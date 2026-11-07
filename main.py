"""Run a short normal and streaming request against the selected provider."""

import asyncio
from pathlib import Path

from pydantic import ValidationError

from unified_llm import ChatMessage
from unified_llm.settings import manager_from_env


async def main() -> None:
    try:
        manager = manager_from_env(Path(__file__).with_name(".env"))
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
