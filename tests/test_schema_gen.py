import types
import pytest
from typing import Optional, Literal
from llm_schema_gen import schema_from_fn, anthropic_schema_from_fn, schemas_from_module


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

def fn_str(name: str) -> str:
    return name

def fn_int(count: int) -> int:
    return count

def fn_float(score: float) -> float:
    return score

def fn_bool(flag: bool) -> bool:
    return flag


def test_str_type():
    s = schema_from_fn(fn_str)
    assert s["parameters"]["properties"]["name"]["type"] == "string"


def test_int_type():
    s = schema_from_fn(fn_int)
    assert s["parameters"]["properties"]["count"]["type"] == "integer"


def test_float_type():
    s = schema_from_fn(fn_float)
    assert s["parameters"]["properties"]["score"]["type"] == "number"


def test_bool_type():
    s = schema_from_fn(fn_bool)
    assert s["parameters"]["properties"]["flag"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# Complex types
# ---------------------------------------------------------------------------

def fn_list(items: list[str]) -> list:
    return items

def fn_dict(mapping: dict[str, int]) -> dict:
    return mapping

def fn_optional(value: Optional[str] = None) -> Optional[str]:
    return value

def fn_literal(color: Literal["red", "blue", "green"]) -> str:
    return color


def test_list_type():
    s = schema_from_fn(fn_list)
    prop = s["parameters"]["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_dict_type():
    s = schema_from_fn(fn_dict)
    prop = s["parameters"]["properties"]["mapping"]
    assert prop["type"] == "object"
    assert prop["additionalProperties"]["type"] == "integer"


def test_optional_not_required():
    s = schema_from_fn(fn_optional)
    assert "value" not in s["parameters"].get("required", [])


def test_literal_enum():
    s = schema_from_fn(fn_literal)
    prop = s["parameters"]["properties"]["color"]
    assert prop["type"] == "string"
    assert set(prop["enum"]) == {"red", "blue", "green"}


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

def fn_required(a: str, b: int) -> None:
    pass

def fn_with_default(a: str, b: int = 0) -> None:
    pass


def test_all_required():
    s = schema_from_fn(fn_required)
    assert "a" in s["parameters"]["required"]
    assert "b" in s["parameters"]["required"]


def test_default_not_required():
    s = schema_from_fn(fn_with_default)
    assert "a" in s["parameters"]["required"]
    assert "b" not in s["parameters"].get("required", [])


# ---------------------------------------------------------------------------
# Name and description
# ---------------------------------------------------------------------------

def fn_named_with_doc(x: int) -> int:
    """Add one to x."""
    return x + 1


def test_schema_name():
    s = schema_from_fn(fn_named_with_doc)
    assert s["name"] == "fn_named_with_doc"


def test_schema_description():
    s = schema_from_fn(fn_named_with_doc)
    assert "Add one" in s["description"]


def test_no_docstring_empty_description():
    def undocumented(x: int) -> int:
        return x
    s = schema_from_fn(undocumented)
    assert s["description"] == ""


# ---------------------------------------------------------------------------
# Google-style docstring param descriptions
# ---------------------------------------------------------------------------

def fn_with_param_docs(city: str, units: str = "metric") -> str:
    """Get weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units, either 'metric' or 'imperial'.
    """
    return ""


def test_param_description_from_docstring():
    s = schema_from_fn(fn_with_param_docs)
    assert "city" in s["parameters"]["properties"]["city"].get("description", "").lower() or \
           "city" in s["parameters"]["properties"]["city"].get("description", "City").lower()


def test_param_description_present():
    s = schema_from_fn(fn_with_param_docs)
    assert s["parameters"]["properties"]["city"]["description"] != ""


# ---------------------------------------------------------------------------
# OpenAI format structure
# ---------------------------------------------------------------------------

def test_openai_top_level_keys():
    def simple(x: str) -> str:
        return x
    s = schema_from_fn(simple)
    assert "name" in s
    assert "description" in s
    assert "parameters" in s


def test_openai_parameters_type():
    def simple(x: str) -> str:
        return x
    s = schema_from_fn(simple)
    assert s["parameters"]["type"] == "object"
    assert "properties" in s["parameters"]


# ---------------------------------------------------------------------------
# Anthropic format
# ---------------------------------------------------------------------------

def test_anthropic_schema_has_input_schema():
    def simple(x: str) -> str:
        return x
    s = anthropic_schema_from_fn(simple)
    assert "input_schema" in s
    assert "parameters" not in s


def test_anthropic_schema_input_schema_type():
    def simple(x: str) -> str:
        return x
    s = anthropic_schema_from_fn(simple)
    assert s["input_schema"]["type"] == "object"


def test_anthropic_schema_name_and_description():
    def my_tool(x: int) -> int:
        """Does something."""
        return x
    s = anthropic_schema_from_fn(my_tool)
    assert s["name"] == "my_tool"
    assert "Does something" in s["description"]


# ---------------------------------------------------------------------------
# schemas_from_module
# ---------------------------------------------------------------------------

def test_schemas_from_module_openai():
    mod = types.ModuleType("testmod")
    def tool_a(x: str) -> str:
        return x
    def tool_b(y: int) -> int:
        return y
    mod.tool_a = tool_a
    mod.tool_b = tool_b
    schemas = schemas_from_module(mod)
    names = {s["name"] for s in schemas}
    assert "tool_a" in names
    assert "tool_b" in names


def test_schemas_from_module_anthropic():
    mod = types.ModuleType("testmod2")
    def my_fn(z: float) -> float:
        return z
    mod.my_fn = my_fn
    schemas = schemas_from_module(mod, anthropic=True)
    assert all("input_schema" in s for s in schemas)


def test_schemas_from_module_skips_private():
    mod = types.ModuleType("testmod3")
    def _private(x: int) -> int:
        return x
    def public_tool(x: int) -> int:
        return x
    mod._private = _private
    mod.public_tool = public_tool
    schemas = schemas_from_module(mod)
    names = {s["name"] for s in schemas}
    assert "public_tool" in names
    assert "_private" not in names


def test_schemas_from_module_skips_classes():
    mod = types.ModuleType("testmod4")
    class MyClass:
        pass
    mod.MyClass = MyClass
    schemas = schemas_from_module(mod)
    names = {s["name"] for s in schemas}
    assert "MyClass" not in names
