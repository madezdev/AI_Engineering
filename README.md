# Unified Async LLM Client

Cliente asíncrono para OpenAI y Anthropic con una interfaz común, validación con Pydantic y respuestas en streaming.

## Requisitos e instalación

Se requiere Python 3.12 y una clave de API del proveedor elegido.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Completa la clave correspondiente en `.env`. El archivo `.env` está excluido de Git.

## Configuración

| Variable | Descripción |
| --- | --- |
| `LLM_PROVIDER` | `openai` o `anthropic` |
| `LLM_MODEL` | Nombre del modelo del proveedor; si se omite, se usa un modelo predeterminado |
| `OPENAI_API_KEY` | Necesaria cuando `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Necesaria cuando `LLM_PROVIDER=anthropic` |

Para Anthropic, cambia `LLM_PROVIDER` a `anthropic` y ajusta `LLM_MODEL` al modelo que tengas disponible.

## Ejecución

```bash
python main.py
```

El script pregunta «¿Qué es la entropía?» una vez con respuesta completa y otra vez en streaming. Cada llamada consume tokens de la API configurada.

Para ejecutar las pruebas locales sin consumir créditos de API:

```bash
python -m unittest discover -s tests -v
```

## Uso desde Python

```python
from clients import AsyncLLMManager
from schemas import ChatMessage, ModelConfig

config = ModelConfig(provider="openai", model="gpt-4o-mini", temperature=0.7, max_tokens=512)
manager = AsyncLLMManager(config, api_key="TU_CLAVE")
messages = [ChatMessage(role="user", content="Hola")]

response = await manager.generate(messages)
if response.error:
    print(response.error.code, response.error.message)
else:
    print(response.content)

async for event in manager.stream(messages):
    if event.error:
        print(event.error.code, event.error.message)
        break
    print(event.text, end="", flush=True)
```

`ModelConfig` valida temperatura de 0 a 2 (hasta 1 para Anthropic) y `max_tokens` mayor que cero. Los errores de autenticación, límite de tasa, conexión y API se devuelven en `ModelResponse.error` o en un `StreamEvent.error`; `retryable` indica si conviene reintentar.
