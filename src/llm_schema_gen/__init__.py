"""llm-schema-gen-py — generate Anthropic/OpenAI tool schemas from Python type hints."""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, Callable, get_type_hints


# ---------------------------------------------------------------------------
# Type → JSON Schema mapping
# ---------------------------------------------------------------------------

def _type_to_schema(tp: Any) -> dict:
    """Recursively convert a Python type annotation to a JSON Schema dict."""
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())

    # Optional[X] → {"type": ...}  (not required, handled at field level)
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {}  # complex union — leave untyped

    # Literal["a", "b"] → {"type": "string", "enum": ["a", "b"]}
    if origin is typing.Literal:
        all_str = all(isinstance(a, str) for a in args)
        return {
            "type": "string" if all_str else _infer_primitive(type(args[0])),
            "enum": list(args),
        }

    # list[T] → {"type": "array", "items": {...}}
    if origin is list:
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _type_to_schema(args[0])
        return schema

    # dict[str, T] → {"type": "object", "additionalProperties": {...}}
    if origin is dict:
        schema = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = _type_to_schema(args[1])
        return schema

    # Primitives
    primitive = _infer_primitive(tp)
    if primitive:
        return {"type": primitive}

    return {}  # unknown / unsupported


def _infer_primitive(tp: Any) -> str | None:
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    return mapping.get(tp)


def _is_optional(tp: Any) -> bool:
    origin = getattr(tp, "__origin__", None)
    if origin is typing.Union:
        return type(None) in tp.__args__
    return False


# ---------------------------------------------------------------------------
# Docstring parsing (Google style)
# ---------------------------------------------------------------------------

def _parse_google_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (summary, {param_name: description}) from a Google-style docstring."""
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    summary_lines: list[str] = []
    param_descs: dict[str, str] = {}

    in_args = False
    current_param: str | None = None
    current_desc_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            if current_param:
                param_descs[current_param] = " ".join(current_desc_lines).strip()
                current_param, current_desc_lines = None, []
            continue
        if in_args and stripped and not stripped.endswith(":") and re.match(r"^[A-Za-z_]\w*\s*:", stripped):
            if current_param:
                param_descs[current_param] = " ".join(current_desc_lines).strip()
            m = re.match(r"^([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if m:
                current_param = m.group(1)
                current_desc_lines = [m.group(2)] if m.group(2) else []
            continue
        if in_args and current_param:
            if stripped and not re.match(r"^[A-Za-z_]\w*\s*:", stripped):
                current_desc_lines.append(stripped)
            elif stripped.endswith(":") and len(stripped) < 20:
                # New section header — stop args parsing
                in_args = False
                if current_param:
                    param_descs[current_param] = " ".join(current_desc_lines).strip()
                    current_param, current_desc_lines = None, []
            continue
        if not in_args:
            summary_lines.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc_lines).strip()

    summary = " ".join(l for l in summary_lines if l)
    return summary, param_descs


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

def schema_from_fn(fn: Callable) -> dict:
    """
    Generate an OpenAI-compatible tool schema from a Python function.

    Returns a dict matching the OpenAI function-calling format::

        {
            "name": "my_func",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    """
    name = fn.__name__
    doc = inspect.getdoc(fn) or ""
    summary, param_descs = _parse_google_docstring(doc)

    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        tp = hints.get(param_name, Any)
        prop = _type_to_schema(tp)
        if param_descs.get(param_name):
            prop["description"] = param_descs[param_name]
        properties[param_name] = prop

        # Required if no default and not Optional
        if param.default is inspect.Parameter.empty and not _is_optional(tp):
            required.append(param_name)

    return {
        "name": name,
        "description": summary,
        "parameters": {
            "type": "object",
            "properties": properties,
            **({"required": required} if required else {}),
        },
    }


def anthropic_schema_from_fn(fn: Callable) -> dict:
    """
    Generate an Anthropic-compatible tool schema from a Python function.

    Returns a dict matching the Anthropic tool format::

        {
            "name": "my_func",
            "description": "...",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    """
    openai = schema_from_fn(fn)
    return {
        "name": openai["name"],
        "description": openai["description"],
        "input_schema": openai["parameters"],
    }


def schemas_from_module(module: Any, *, anthropic: bool = False) -> list[dict]:
    """
    Extract tool schemas from all public callables in a module.

    Args:
        module: A Python module object.
        anthropic: If True, return Anthropic-format schemas; else OpenAI format.

    Returns:
        A list of schema dicts.
    """
    gen = anthropic_schema_from_fn if anthropic else schema_from_fn
    results = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and not isinstance(obj, type):
            try:
                results.append(gen(obj))
            except Exception:
                pass
    return results


__all__ = [
    "schema_from_fn",
    "anthropic_schema_from_fn",
    "schemas_from_module",
    "SideEffect",
]
