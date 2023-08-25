from __future__ import annotations

import json
from typing import Any

import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value


class JSONSpliceDouble(Value):
    """JSON splice double values."""

    arg = rlz.one_of([rlz.json, rlz.string])
    start = rlz.integer
    length = rlz.integer
    values = rlz.tuple_of(rlz.strict_numeric, min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_splice_double(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.FloatingValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice double values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : doubles
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceDouble(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_double = json_splice_double
ir.StringValue.json_splice_double = json_splice_double


class JSONSpliceString(Value):
    """JSON splice string values."""

    arg = rlz.one_of([rlz.json, rlz.string])
    start = rlz.integer
    length = rlz.integer
    values = rlz.tuple_of(rlz.string, min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_splice_string(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice double values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : strings
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceString(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_string = json_splice_string
ir.StringValue.json_splice_string = json_splice_string


class JSONSpliceJSON(Value):
    """JSON splice JSON values."""

    arg = rlz.one_of([rlz.json, rlz.string])
    start = rlz.integer
    length = rlz.integer
    values = rlz.tuple_of(rlz.one_of([rlz.json, rlz.string]), min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_splice_json(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.JSONValue | ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice JSON values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : strings
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceJSON(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_json = json_splice_json
ir.StringValue.json_splice_json = json_splice_json


class JSONSplice(Value):
    """JSON splice values."""

    arg = rlz.one_of([rlz.json, rlz.string])
    start = rlz.integer
    length = rlz.integer
    values = rlz.tuple_of(rlz.any, min_length=1)

    output_dtype = rlz.dtype_like('arg')
    output_shape = rlz.shape_like('arg')


def json_splice(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.Value | Any,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice JSON values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : Any
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    items = tuple(values)
    if values and not isinstance(values[0], ir.Value):
        items = tuple([json.dumps(x) for x in items])
    return JSONSpliceJSON(arg, start, length, items).to_expr()


ir.JSONValue.json_splice = json_splice
ir.StringValue.json_splice = json_splice
