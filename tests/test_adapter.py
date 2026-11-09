# ABOUTME: Tests for the Anthropic provider adapter's request assembly, response
# ABOUTME: extraction, and error mapping (T3 red).

from types import SimpleNamespace

import pytest

import anthropic
from scientist.adapter import AdapterError, AnthropicAdapter


class FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self):
        self.calls = []
        self._response = None

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self._response)


class FakeClient:
    def __init__(self, response=None):
        self.messages = FakeMessages()
        self.messages._response = response


def make_text_block(text):
    return SimpleNamespace(type="text", text=text)


def make_nontext_block(block_type):
    return SimpleNamespace(type=block_type, text="should-be-ignored")


def make_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def test_adapter_sends_model_prompt_and_max_tokens():
    fake_client = FakeClient(response=make_response([make_text_block("ok")]))

    adapter = AnthropicAdapter(client=fake_client, max_tokens=1234)
    adapter.propose("my prompt", "claude-x")

    assert len(fake_client.messages.calls) == 1
    recorded = fake_client.messages.calls[0]
    assert recorded == {
        "model": "claude-x",
        "max_tokens": 1234,
        "messages": [{"role": "user", "content": "my prompt"}],
    }
    assert "thinking" not in recorded


def test_adapter_concatenates_text_blocks():
    content = [
        make_text_block("block-A-"),
        make_nontext_block("tool_use"),
        make_text_block("block-B-"),
        make_nontext_block("custom-sentinel"),
        make_text_block("block-C-"),
    ]
    fake_client = FakeClient(response=make_response(content))

    adapter = AnthropicAdapter(client=fake_client, max_tokens=100)
    result = adapter.propose("concat prompt", "claude-concat")

    assert result == "block-A-block-B-block-C-"


def test_adapter_refusal_raises_adapter_error():
    fake_client = FakeClient(
        response=make_response([make_text_block("I can't help")], stop_reason="refusal")
    )

    adapter = AnthropicAdapter(client=fake_client, max_tokens=100)
    with pytest.raises(AdapterError):
        adapter.propose("refusal prompt", "claude-refusal")


def test_adapter_empty_text_raises_adapter_error():
    fake_client = FakeClient(response=make_response([]))

    adapter = AnthropicAdapter(client=fake_client, max_tokens=100)
    with pytest.raises(AdapterError):
        adapter.propose("empty prompt", "claude-empty")


def test_adapter_empty_text_error_reports_stop_reason_and_block_types():
    content = [make_nontext_block("thinking")]
    fake_client = FakeClient(
        response=make_response(content, stop_reason="max_tokens")
    )

    adapter = AnthropicAdapter(client=fake_client, max_tokens=100)
    with pytest.raises(AdapterError) as exc_info:
        adapter.propose("empty prompt", "claude-empty")

    message = str(exc_info.value)
    assert "stop_reason=max_tokens" in message
    assert "thinking" in message


def test_adapter_sdk_error_chains_to_adapter_error():
    original = anthropic.APIError(
        "boom",
        SimpleNamespace(method="GET", url="http://example.com"),
        body=None,
    )

    class RaisingMessages:
        def stream(self, **kwargs):
            raise original

    class RaisingClient:
        messages = RaisingMessages()

    adapter = AnthropicAdapter(client=RaisingClient(), max_tokens=100)
    with pytest.raises(AdapterError) as exc_info:
        adapter.propose("error prompt", "claude-error")

    assert exc_info.value.__cause__ is original


def test_adapter_stream_final_message_error_chains_to_adapter_error():
    original = anthropic.APIError(
        "stream broke",
        SimpleNamespace(method="GET", url="http://example.com"),
        body=None,
    )

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_final_message(self):
            raise original

    class BrokenMessages:
        def stream(self, **kwargs):
            return BrokenStream()

    class BrokenClient:
        messages = BrokenMessages()

    adapter = AnthropicAdapter(client=BrokenClient(), max_tokens=100)
    with pytest.raises(AdapterError) as exc_info:
        adapter.propose("error prompt", "claude-error")

    assert exc_info.value.__cause__ is original


def test_adapter_default_client_from_env(monkeypatch):
    sentinel = object()

    def fake_constructor(**kwargs):
        return sentinel

    monkeypatch.setattr(anthropic, "Anthropic", fake_constructor)

    adapter = AnthropicAdapter()

    assert adapter._client is sentinel


def test_adapter_never_retries_internally():
    class CountingMessages:
        def __init__(self):
            self.call_count = 0

        def stream(self, **kwargs):
            self.call_count += 1
            raise anthropic.APIError(
                "always fails",
                SimpleNamespace(method="GET", url="http://example.com"),
                body=None,
            )

    class CountingClient:
        messages = CountingMessages()

    adapter = AnthropicAdapter(client=CountingClient(), max_tokens=100)
    with pytest.raises(AdapterError):
        adapter.propose("retry prompt", "claude-retry")

    assert CountingClient.messages.call_count == 1
