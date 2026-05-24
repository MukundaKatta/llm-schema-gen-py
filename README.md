# llm-schema-gen-py

Generate Anthropic/OpenAI tool schemas from Python type hints and Google-style docstrings.

```bash
pip install llm-schema-gen-py
```

## Quick start

```python
from typing import Optional, Literal
from llm_schema_gen import schema_from_fn, anthropic_schema_from_fn

def get_weather(city: str, units: Literal["metric", "imperial"] = "metric") -> str:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units to use.
    """
    ...

# OpenAI format
schema = schema_from_fn(get_weather)
# {
#   "name": "get_weather",
#   "description": "Get current weather for a city.",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "city": {"type": "string", "description": "The city name to look up."},
#       "units": {"type": "string", "enum": ["metric", "imperial"]}
#     },
#     "required": ["city"]
#   }
# }

# Anthropic format (input_schema instead of parameters)
schema = anthropic_schema_from_fn(get_weather)
```

## Supported types

| Python | JSON Schema |
|--------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[T]` | `{"type": "array", "items": ...}` |
| `dict[str, T]` | `{"type": "object", "additionalProperties": ...}` |
| `Optional[T]` | type of T, not in `required` |
| `Literal["a", "b"]` | `{"type": "string", "enum": ["a", "b"]}` |

## Batch from module

```python
import my_tools
from llm_schema_gen import schemas_from_module

schemas = schemas_from_module(my_tools)                    # OpenAI
schemas = schemas_from_module(my_tools, anthropic=True)    # Anthropic
```

## Zero dependencies
