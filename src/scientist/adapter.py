# ABOUTME: LLM provider adapter protocol and Anthropic SDK wrapper -- sends prompts and
# ABOUTME: returns reply text, surfacing API failures and refusals as transient errors.

"""Provider adapter for LLM backends.

The Anthropic adapter is deliberately thin and model-agnostic:
* it passes the caller's model name through verbatim,
* it sends a single user message with no model-specific ``thinking`` configuration,
* it converts the SDK response (or any SDK failure) into the generic
  :class:`AdapterError` that the scientist loop treats as transient.

This keeps per-run model selection free and avoids binding the adapter to one
model family.
"""

from typing import Protocol

import anthropic


class AdapterError(Exception):
    """Raised when an adapter cannot produce a proposal text."""


class ProviderAdapter(Protocol):
    """Spec interface: one method, no model-specific configuration."""

    def propose(self, prompt: str, model: str) -> str: ...


class AnthropicAdapter:
    """Thin wrapper around the official ``anthropic`` SDK.

    The constructor accepts an injectable client for testing; when omitted, a
    default ``anthropic.Anthropic()`` is built from the environment
    (``ANTHROPIC_API_KEY``).
    """

    def __init__(
        self,
        client: "anthropic.Anthropic | None" = None,
        max_tokens: int = 2000,
    ) -> None:
        self._client = client if client is not None else anthropic.Anthropic()
        self._max_tokens = max_tokens

    def propose(self, prompt: str, model: str) -> str:
        """Send ``prompt`` to ``model`` and return the concatenated text blocks.

        Raises:
            AdapterError: if the SDK raises any exception, if the response's
                ``stop_reason`` is ``"refusal"``, or if no text blocks are present.
        """
        try:
            with self._client.messages.stream(
                model=model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()
        except Exception as error:
            raise AdapterError(
                f"Anthropic API call failed: {error}"
            ) from error

        if response.stop_reason == "refusal":
            raise AdapterError("Model refused the request")

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        if not text:
            block_types = [block.type for block in response.content]
            raise AdapterError(
                "Response contained no text blocks "
                f"(stop_reason={response.stop_reason}, blocks={block_types})"
            )

        return text
