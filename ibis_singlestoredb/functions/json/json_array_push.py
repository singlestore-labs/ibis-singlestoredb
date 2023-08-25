from __future__ import annotations

import json
from typing import Any

import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONArrayPushDouble(Value):
    """JSON array append."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.strict_numeric

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_array_push_double(
    arg: ir.JSONValue | ir.StringValue, value: ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Append a double to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : float
        Float value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushDouble(arg, value).to_expr()


ir.JSONValue.json_array_push_double = json_array_push_double
ir.StringValue.json_array_push_double = json_array_push_double


class JSONArrayPushString(Value):
    """JSON array append."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.string

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_array_push_string(
    arg: ir.JSONValue | ir.StringValue, value: ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Append a string to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        String value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushString(arg, value).to_expr()


ir.JSONValue.json_array_push_string = json_array_push_string
ir.StringValue.json_array_push_string = json_array_push_string


class JSONArrayPushJSON(Value):
    """JSON array append."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.one_of([rlz.json, rlz.string])

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_array_push_json(
    arg: ir.JSONValue | ir.StringValue, value: ir.JSONValue | ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Append a JSON object to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : JSON or string
        JSON value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushJSON(arg, value).to_expr()


ir.JSONValue.json_array_push_json = json_array_push_json
ir.StringValue.json_array_push_json = json_array_push_json


class JSONArrayPush(Value):
    """JSON array append."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.any

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_array_push(
    arg: ir.JSONValue | ir.StringValue, value: ir.Value | Any,
) -> ir.JSONValue | ir.StringValue:
    """
    Append a JSON object to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : Any
        Value to append

    Returns
    -------
    JSON or string column

    """
    if not isinstance(value, ir.Value):
        value = json.dumps(value)
    return JSONArrayPushJSON(arg, value).to_expr()


ir.JSONValue.json_array_push = json_array_push
ir.StringValue.json_array_push = json_array_push
