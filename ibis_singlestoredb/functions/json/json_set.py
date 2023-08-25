from __future__ import annotations

import json
from typing import Any

import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONSetDouble(Value):
    """JSON set double value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)
    value = rlz.double

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_set_double(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a double value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON object
    *key_value : strings or ints and double
        Keys or zero-indexed array positions followed by the double value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetDouble(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_double = json_set_double
ir.StringValue.json_set_double = json_set_double


class JSONSetString(Value):
    """JSON set string value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)
    value = rlz.string

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_set_string(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a double value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON object
    *key_value : strings or ints
        Keys or zero-indexed array positions followed by the string value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetString(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_string = json_set_string
ir.StringValue.json_set_string = json_set_string


class JSONSetJSON(Value):
    """JSON set JSON value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)
    value = rlz.string

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_set_json(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a JSON value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON object
    *key_value : strings or ints
        Keys or zero-indexed array positions followed by the JSON value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetJSON(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_json = json_set_json
ir.StringValue.json_set_json = json_set_json


class JSONSet(Value):
    """JSON set any value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)
    value = rlz.any

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_set(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue | Any,
) -> ir.JSONValue | ir.StringValue:
    """
    Set any value at the specified key path.

    Parameters
    ----------
    arg : Any
        Any object that can be persisted to a JSON string
    *key_value : strings or ints
        Keys or zero-indexed array positions followed by the JSON value

    Returns
    -------
    String or JSON Column

    """
    if not isinstance(key_value[-1], ir.Value):
        value = json.dumps(key_value[-1])
    return JSONSetJSON(arg, key_value[:-1], value).to_expr()


ir.JSONValue.json_set = json_set
ir.StringValue.json_set = json_set
