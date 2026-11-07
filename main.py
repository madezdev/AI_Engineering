"""Run a short normal and streaming request against the selected provider."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from clients import AsyncLLMManager
from schemas import ChatMessage, ModelConfig


async def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL") or (
        "gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest"
    )
    try:
        config = ModelConfig(provider=provider, model=model)
        key_name = "OPENAI_API_KEY" if config.provider == "openai" else "ANTHROPIC_API_KEY"
        manager = AsyncLLMManager(config, os.getenv(key_name, ""))
    except (ValidationError, ValueError) as exc:
        print(f"Configuración inválida: {exc}")
        return

    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    response = await manager.generate(messages)
    print("Respuesta normal:")
    if response.error:
        print(f"Error ({response.error.code}): {response.error.message}")
    else:
        print(response.content)

    print("\nRespuesta en streaming:")
    async for event in manager.stream(messages):
        if event.error:
            print(f"\nError ({event.error.code}): {event.error.message}")
            break
        print(event.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
