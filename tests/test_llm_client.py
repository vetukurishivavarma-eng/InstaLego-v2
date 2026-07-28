import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from landtitle.llm.client import QwenClient


def _mock_response(content: str):
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return mock


def test_missing_base_url_raises_clearly():
    with pytest.raises(RuntimeError, match="LLM_API_BASE_URL"):
        QwenClient(base_url=None)
    with pytest.raises(RuntimeError, match="LLM_API_BASE_URL"):
        QwenClient(base_url="")


@patch("landtitle.llm.client.requests.post")
def test_generate_sends_expected_request_shape(mock_post):
    mock_post.return_value = _mock_response("hello back")
    client = QwenClient(base_url="https://outflank-filler-bullwhip.ngrok-free.dev", api_key=None)

    result = client.generate("system prompt", "user prompt", temperature=0.1, max_new_tokens=2048)

    assert result == "hello back"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    # URL is built as {base_url}/v1/chat/completions — base_url itself must
    # NOT already include "/v1" (it's the bare ngrok origin).
    assert args[0] == "https://outflank-filler-bullwhip.ngrok-free.dev/v1/chat/completions"
    assert kwargs["json"] == {
        "messages": [{"role": "user", "content": "system prompt\n\nuser prompt"}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    assert kwargs["timeout"] == 120.0


@patch("landtitle.llm.client.requests.post")
def test_generate_sends_ngrok_skip_header(mock_post):
    mock_post.return_value = _mock_response("ok")
    client = QwenClient(base_url="https://example.ngrok-free.dev")
    client.generate("sys", "user")
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["ngrok-skip-browser-warning"] == "true"


@patch("landtitle.llm.client.requests.post")
def test_generate_omits_authorization_header_when_no_api_key(mock_post):
    mock_post.return_value = _mock_response("ok")
    client = QwenClient(base_url="https://example.test", api_key=None)
    client.generate("sys", "user")
    _, kwargs = mock_post.call_args
    assert "Authorization" not in kwargs["headers"]


@patch("landtitle.llm.client.requests.post")
def test_generate_includes_authorization_header_when_api_key_set(mock_post):
    mock_post.return_value = _mock_response("ok")
    client = QwenClient(base_url="https://example.test", api_key="secret")
    client.generate("sys", "user")
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


@patch("landtitle.llm.client.requests.post")
def test_generate_strips_whitespace_from_response(mock_post):
    mock_post.return_value = _mock_response("  padded response  \n")
    client = QwenClient(base_url="https://example.test")
    assert client.generate("sys", "user") == "padded response"


@patch("landtitle.llm.client.requests.post")
def test_generate_raises_on_connection_failure(mock_post):
    import requests

    mock_post.side_effect = requests.exceptions.ConnectionError("refused")
    client = QwenClient(base_url="https://example.test")
    with pytest.raises(RuntimeError, match="example.test"):
        client.generate("sys", "user")


@patch("landtitle.llm.client.requests.post")
def test_generate_raises_on_timeout(mock_post):
    import requests

    mock_post.side_effect = requests.exceptions.Timeout("timed out")
    client = QwenClient(base_url="https://example.test")
    with pytest.raises(RuntimeError, match="did not respond within"):
        client.generate("sys", "user")


@patch("landtitle.llm.client.requests.post")
def test_generate_raises_on_non_json_response(mock_post):
    mock = MagicMock()
    mock.json.side_effect = ValueError("no JSON")
    mock.text = "<html>ngrok warning page</html>"
    mock_post.return_value = mock
    client = QwenClient(base_url="https://example.test")
    with pytest.raises(RuntimeError, match="did not return valid JSON"):
        client.generate("sys", "user")


@patch("landtitle.llm.client.requests.post")
def test_generate_raises_on_unexpected_response_shape(mock_post):
    mock = MagicMock()
    mock.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock
    client = QwenClient(base_url="https://example.test")
    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        client.generate("sys", "user")


class _Dummy(BaseModel):
    value: str


@patch("landtitle.llm.client.requests.post")
def test_extract_structured_parses_json_from_response(mock_post):
    mock_post.return_value = _mock_response(json.dumps({"value": "ok"}))
    client = QwenClient(base_url="https://example.test")
    result = client.extract_structured("sys", "user", _Dummy)
    assert result.value == "ok"


@patch("landtitle.llm.client.requests.post")
def test_extract_structured_retries_on_invalid_json(mock_post):
    mock_post.side_effect = [
        _mock_response("not json at all"),
        _mock_response(json.dumps({"value": "recovered"})),
    ]
    client = QwenClient(base_url="https://example.test")
    result = client.extract_structured("sys", "user", _Dummy, max_retries=1)
    assert result.value == "recovered"
    assert mock_post.call_count == 2
