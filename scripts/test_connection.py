"""Quick standalone check that LLM_API_BASE_URL is reachable and returns the
expected response shape, before running the full pipeline against real
documents.

Usage:
    python scripts/test_connection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel

from landtitle.llm.client import QwenClient


class _Ping(BaseModel):
    status: str
    number: int


def main() -> None:
    client = QwenClient()
    print(f"Testing connection to {client.base_url} ...")

    print("\n[1/2] Raw generate() call...")
    reply = client.generate(
        system_prompt="You are a test assistant.",
        user_prompt="Reply with exactly the words: connection successful",
        max_new_tokens=20,
    )
    print(f"Response: {reply!r}")

    print("\n[2/2] Structured JSON extraction (extract_structured) call...")
    result = client.extract_structured(
        system_prompt="You are a test assistant that only outputs JSON.",
        user_prompt='Set status to "ok" and number to 42.',
        schema=_Ping,
    )
    print(f"Parsed result: {result!r}")

    print("\nBoth calls succeeded — the model server is reachable and returning "
          "the expected response shape. Safe to run the full pipeline now.")


if __name__ == "__main__":
    main()
