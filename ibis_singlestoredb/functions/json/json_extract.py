from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONExtractDouble(Value):
    """JSON extract double value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)

    output_dtype = dt.double
    output_shape = rlz.shape_like('arg')


def json_extract_double(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.FloatingValue:
    """
    Extract a double value from a JSON object.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    Double

    """
    return JSONExtractDouble(arg, key_path).to_expr()


ir.JSONValue.json_extract_double = json_extract_double
ir.StringValue.json_extract_double = json_extract_double


class JSONExtractString(Value):
    """JSON extract string value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


def json_extract_string(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.StringValue:
    """
    Extract a string value from a JSON object.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    String

    """
    return JSONExtractString(arg, key_path).to_expr()


ir.JSONValue.json_extract_string = json_extract_string
ir.StringValue.json_extract_string = json_extract_string


class JSONExtractJSON(Value):
    """JSON extract string value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_extract_json(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Extract a JSON value from a JSON object.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    JSON or string

    """
    return JSONExtractJSON(arg, key_path).to_expr()


ir.JSONValue.json_extract_json = json_extract_json
ir.StringValue.json_extract_json = json_extract_json


class JSONExtractBigint(Value):
    """JSON extract integer value."""

    arg = rlz.one_of([rlz.json, rlz.string])
    key_path = rlz.tuple_of(rlz.one_of([rlz.string, rlz.integer]), min_length=1)

    output_dtype = dt.int64
    output_shape = rlz.shape_like('arg')


def json_extract_bigint(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.IntegerValue:
    """
    Extract an int value from a JSON object.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    Integer

    """
    return JSONExtractBigint(arg, key_path).to_expr()


ir.JSONValue.json_extract_bigint = json_extract_bigint
ir.StringValue.json_extract_bigint = json_extract_bigint
