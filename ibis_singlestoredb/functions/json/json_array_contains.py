from __future__ import annotations

import json
from typing import Any

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONArrayContainsDouble(Value):
    """JSON array element test."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.strict_numeric

    output_dtype = dt.bool
    output_shape = rlz.shape_like('arg')


def json_array_contains_double(
    arg: ir.JSONValue | ir.StringValue, value: ir.NumericValue,
) -> ir.BooleanValue:
    """
    Does the array contain the given float value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : float
        Float value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsDouble(arg, value).to_expr()


ir.JSONValue.json_array_contains_double = json_array_contains_double
ir.StringValue.json_array_contains_double = json_array_contains_double


class JSONArrayContainsString(Value):
    """JSON array element test."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.string

    output_dtype = dt.bool
    output_shape = rlz.shape_like('arg')


def json_array_contains_string(
    arg: ir.JSONValue | ir.StringValue, value: ir.StringValue,
) -> ir.BooleanValue:
    """
    Does the array contain the given string value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        String value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsString(arg, value).to_expr()


ir.JSONValue.json_array_contains_string = json_array_contains_string
ir.StringValue.json_array_contains_string = json_array_contains_string


class JSONArrayContainsJSON(Value):
    """JSON array element test."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.one_of([rlz.json, rlz.string])

    output_dtype = dt.bool
    output_shape = rlz.shape_like('arg')


def json_array_contains_json(
    arg: ir.JSONValue | ir.StringValue, value: ir.JSONValue | ir.StringValue,
) -> ir.BooleanValue:
    """
    Does the array contain the given JSON value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        JSON value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsJSON(arg, value).to_expr()


ir.JSONValue.json_array_contains_json = json_array_contains_json
ir.StringValue.json_array_contains_json = json_array_contains_json


class JSONArrayContains(Value):
    """JSON array element test."""

    arg = rlz.one_of([rlz.json, rlz.string])
    value = rlz.any

    output_dtype = dt.bool
    output_shape = rlz.shape_like('arg')


def json_array_contains(
    arg: ir.JSONValue | ir.StringValue, value: ir.Value | Any,
) -> ir.BooleanValue:
    """
    Does the array contain the given value?

    Parameters
    ----------
    arg : JSON or string
        JSON object
    value : Any
        Value to search for

    Returns
    -------
    Boolean column

    """
    if not isinstance(value, ir.Value):
        value = json.dumps(value)
    return JSONArrayContainsJSON(arg, value).to_expr()


ir.JSONValue.json_array_contains = json_array_contains
ir.StringValue.json_array_contains = json_array_contains
