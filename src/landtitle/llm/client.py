"""QwenClient — HTTP client for a remote model server, not an in-process model.

Holds no model weights and does no local inference. Every call POSTs to
{LLM_API_BASE_URL}/v1/chat/completions and expects back
`{"choices": [{"message": {"role": "assistant", "content": "..."}}]}`.
The model (Qwen2.5-7B-Instruct, 4-bit quantized) runs wherever that URL
points — e.g. a Kaggle notebook running a FastAPI server, tunneled via
ngrok/localtunnel. Do not swap the model name or add fine-tuning without
re-running the manual validation described in the project README: 14B was
tested side-by-side on the same tasks and produced identical errors,
confirming 7B is sufficient and that errors are a prompting/architecture
problem, not a model-capability problem.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from landtitle.config import (
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_API_TIMEOUT,
    LLM_MAX_NEW_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TRANSIENT_RETRY_ATTEMPTS,
    LLM_TRANSIENT_RETRY_BACKOFF_SECONDS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class QwenClient:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        base_url: str | None = LLM_API_BASE_URL,
        api_key: str | None = LLM_API_KEY,
        timeout: float = LLM_API_TIMEOUT,
        seed: int | None = LLM_SEED,
    ):
        if not base_url:
            raise RuntimeError(
                "LLM_API_BASE_URL is not set. Point it at your model server's base URL "
                "(e.g. LLM_API_BASE_URL=https://your-ngrok-id.ngrok-free.dev) before "
                "constructing QwenClient — there is no local fallback."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.seed = seed

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        content = f"{system_prompt}\n\n{user_prompt}" if system_prompt and system_prompt.strip() else user_prompt
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if self.seed is not None:
            # Confirmed live: the server tolerates an unrecognized field
            # without error. Whether it actually increases determinism
            # depends on the server's own generation code applying it -- see
            # config.py's LLM_SEED docstring.
            payload["seed"] = self.seed
        headers = {
            "Content-Type": "application/json",
            # ngrok's free tier serves an HTML "visit site" interstitial to
            # anonymous requests unless this header is present — without it,
            # response.json() below fails on that HTML page with a confusing
            # error that has nothing to do with the actual model server.
            "ngrok-skip-browser-warning": "true",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "POST %s (prompt=%d chars, max_new_tokens=%d)", url, len(content), max_new_tokens
        )
        start = time.monotonic()
        response = None
        attempt = 0
        while True:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout as exc:
                # Confirmed live: a real call timed out at exactly the (then-120s) limit
                # while a second concurrent request was likely queued behind it on the
                # user's single Kaggle server -- observed real latencies of 47-113s per
                # call even without contention, so a bare timeout here isn't necessarily
                # a dead server, and retrying once or twice is worth it before giving up.
                if attempt < LLM_TRANSIENT_RETRY_ATTEMPTS:
                    attempt += 1
                    logger.warning(
                        "LLM endpoint at %s did not respond within %.0fs (attempt %d/%d), retrying in %.0fs",
                        url, self.timeout, attempt, LLM_TRANSIENT_RETRY_ATTEMPTS,
                        LLM_TRANSIENT_RETRY_BACKOFF_SECONDS,
                    )
                    time.sleep(LLM_TRANSIENT_RETRY_BACKOFF_SECONDS)
                    continue
                raise RuntimeError(
                    f"LLM endpoint at {url} did not respond within {self.timeout}s after "
                    f"{attempt + 1} attempt(s). It may be overloaded, or the Kaggle session "
                    f"behind it may have stalled or restarted."
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(
                    f"Could not reach the LLM endpoint at {url}. The server or ngrok tunnel may be "
                    f"down, or LLM_API_BASE_URL may be stale (ngrok free-tier URLs change every "
                    f"time the Kaggle session restarts)."
                ) from exc
            except requests.exceptions.HTTPError as exc:
                body_preview = response.text[:500] if response is not None else ""
                # Only a 5xx is retried -- confirmed live, the user's own
                # Kaggle FastAPI server returned one transient 500 while the
                # model was still loading, which succeeded on retry with no
                # other change. A 4xx means the request itself is wrong
                # (bad payload, auth) and retrying it just repeats the same
                # failure.
                is_server_error = response is not None and 500 <= response.status_code < 600
                if is_server_error and attempt < LLM_TRANSIENT_RETRY_ATTEMPTS:
                    attempt += 1
                    logger.warning(
                        "LLM endpoint at %s returned HTTP %d (attempt %d/%d), retrying in %.0fs: %s",
                        url, response.status_code, attempt, LLM_TRANSIENT_RETRY_ATTEMPTS,
                        LLM_TRANSIENT_RETRY_BACKOFF_SECONDS, body_preview,
                    )
                    time.sleep(LLM_TRANSIENT_RETRY_BACKOFF_SECONDS)
                    continue
                raise RuntimeError(
                    f"LLM endpoint at {url} returned HTTP {response.status_code}: {body_preview}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"LLM request to {url} failed: {exc}") from exc

        logger.info("Response received in %.1fs", time.monotonic() - start)

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"LLM endpoint at {url} did not return valid JSON (got: {response.text[:500]!r}). "
                f"If this is an ngrok URL, confirm the tunnel points at your FastAPI server and "
                f"not an ngrok landing/warning page."
            ) from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape from {url}: {data}") from exc

    def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_retries: int = 2,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    ) -> T:
        """Ask the model for JSON matching `schema`, validate, and retry with
        an explicit correction request if parsing/validation fails."""
        schema_prompt = (
            f"{user_prompt}\n\n"
            f"Respond with ONLY a single JSON object matching this schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema.model_json_schema())}"
        )
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            raw = self.generate(system_prompt, schema_prompt, max_new_tokens=max_new_tokens)
            match = _JSON_BLOCK_RE.search(raw)
            candidate = match.group(0) if match else raw
            try:
                data = json.loads(candidate)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "extract_structured(%s) failed to parse/validate on attempt %d/%d: %s",
                    schema.__name__, attempt + 1, max_retries + 1, exc,
                )
                schema_prompt = (
                    f"Your previous response could not be parsed as valid JSON matching the "
                    f"schema. Error: {exc}\n\nOriginal request:\n{user_prompt}\n\n"
                    f"Respond with ONLY a corrected JSON object matching this schema:\n"
                    f"{json.dumps(schema.model_json_schema())}"
                )
        raise ValueError(f"Failed to extract valid {schema.__name__} after {max_retries + 1} attempts") from last_error
