"""Async OpenAI and Anthropic adapters with a common interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamEvent


def _error_from_exception(exc: Exception) -> LLMError:
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return LLMError(code="rate_limit", message="Límite de solicitudes alcanzado.", retryable=True)
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return LLMError(code="authentication", message="La clave de API no es válida.")
    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ),
    ):
        return LLMError(code="network", message="No se pudo conectar con el proveedor.", retryable=True)
    return LLMError(code="api_error", message="El proveedor rechazó la solicitud.")


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Return a complete response or a structured API error."""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        """Yield text fragments, or one structured error event."""


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if config.provider != "openai":
            raise ValueError("OpenAIClient requiere provider='openai'")
        super().__init__(config)
        self._client = AsyncOpenAI(api_key=api_key)

    def _parameters(self, messages: Sequence[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [message.model_dump() for message in messages],
        }

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(**self._parameters(messages))
            content = response.choices[0].message.content or ""
            return ModelResponse(provider="openai", model=self.config.model, content=content)
        except openai.APIError as exc:
            return ModelResponse(
                provider="openai", model=self.config.model, error=_error_from_exception(exc)
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            response = await self._client.chat.completions.create(
                **self._parameters(messages), stream=True
            )
            async for chunk in response:
                if chunk.choices:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield StreamEvent(text=text)
        except openai.APIError as exc:
            yield StreamEvent(error=_error_from_exception(exc))


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if config.provider != "anthropic":
            raise ValueError("AnthropicClient requiere provider='anthropic'")
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=api_key)

    def _parameters(self, messages: Sequence[ChatMessage]) -> dict:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        conversation = [
            message.model_dump() for message in messages if message.role != "system"
        ]
        parameters = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": conversation,
        }
        if system:
            parameters["system"] = system
        return parameters

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._parameters(messages))
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return ModelResponse(provider="anthropic", model=self.config.model, content=content)
        except anthropic.APIError as exc:
            return ModelResponse(
                provider="anthropic", model=self.config.model, error=_error_from_exception(exc)
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        try:
            async with self._client.messages.stream(**self._parameters(messages)) as response:
                async for text in response.text_stream:
                    if text:
                        yield StreamEvent(text=text)
        except anthropic.APIError as exc:
            yield StreamEvent(error=_error_from_exception(exc))


class AsyncLLMManager:
    """Select the provider while keeping one generate/stream API."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        if not api_key:
            raise ValueError("Se requiere la clave de API del proveedor")
        self.client: BaseLLMClient
        if config.provider == "openai":
            self.client = OpenAIClient(config, api_key)
        else:
            self.client = AnthropicClient(config, api_key)

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        return self.client.stream(messages)
