from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class JSONArrayUnpack(Value):
    """32-bit float JSON array unpacker."""

    arg = binary

    output_dtype = dt.json
    output_shape = rlz.shape_like('arg')


def json_array_unpack(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 32-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpack(arg).to_expr()


ir.BinaryValue.json_array_unpack = json_array_unpack


class JSONArrayUnpackI8(JSONArrayUnpack):
    """8-bit integer JSON array unpacker."""


def json_array_unpack_i8(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing an 8-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackI8(arg).to_expr()


ir.BinaryValue.json_array_unpack_i8 = json_array_unpack_i8


class JSONArrayUnpackI16(JSONArrayUnpack):
    """16-bit integer JSON array unpacker."""


def json_array_unpack_i16(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 16-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackI16(arg).to_expr()


ir.BinaryValue.json_array_unpack_i16 = json_array_unpack_i16


class JSONArrayUnpackI32(JSONArrayUnpack):
    """32-bit integer JSON array unpacker."""


def json_array_unpack_i32(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 32-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackI32(arg).to_expr()


ir.BinaryValue.json_array_unpack_i32 = json_array_unpack_i32


class JSONArrayUnpackI64(JSONArrayUnpack):
    """64-bit integer JSON array unpacker."""


def json_array_unpack_i64(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 64-bit int vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackI64(arg).to_expr()


ir.BinaryValue.json_array_unpack_i64 = json_array_unpack_i64


class JSONArrayUnpackF32(JSONArrayUnpack):
    """32-bit float JSON array unpacker."""


def json_array_unpack_f32(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 32-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackF32(arg).to_expr()


ir.BinaryValue.json_array_unpack_f32 = json_array_unpack_f32


class JSONArrayUnpackF64(JSONArrayUnpack):
    """64-bit float JSON array unpacker."""


def json_array_unpack_f64(arg: ir.BinaryValue) -> ir.JSONValue:
    """
    Convert an encoded blob representing a 64-bit float vector to a JSON array.

    Parameters
    ----------
    arg : blob
        Encoded array

    Returns
    -------
    JSON column

    """
    return JSONArrayUnpackF64(arg).to_expr()


ir.BinaryValue.json_array_unpack_f64 = json_array_unpack_f64
