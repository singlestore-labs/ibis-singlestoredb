from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class JSONArrayPack(Value):
    """32-bit float JSON array packer."""

    arg = rlz.one_of([rlz.string, rlz.json])

    output_dtype = dt.binary
    output_shape = rlz.shape_like('arg')


def json_array_pack(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 32-bit floats to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPack(arg).to_expr()


ir.StringValue.json_array_pack = json_array_pack
ir.JSONValue.json_array_pack = json_array_pack


class JSONArrayPackI8(JSONArrayPack):
    """8-bit integer JSON array packer."""


def json_array_pack_i8(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 8-bit ints to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackI8(arg).to_expr()


ir.StringValue.json_array_pack_i8 = json_array_pack_i8
ir.JSONValue.json_array_pack_i8 = json_array_pack_i8


class JSONArrayPackI16(JSONArrayPack):
    """16-bit integer JSON array packer."""


def json_array_pack_i16(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 16-bit ints to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackI16(arg).to_expr()


ir.StringValue.json_array_pack_i16 = json_array_pack_i16
ir.JSONValue.json_array_pack_i16 = json_array_pack_i16


class JSONArrayPackI32(JSONArrayPack):
    """32-bit integer JSON array packer."""


def json_array_pack_i32(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 32-bit ints to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackI32(arg).to_expr()


ir.StringValue.json_array_pack_i32 = json_array_pack_i32
ir.JSONValue.json_array_pack_i32 = json_array_pack_i32


class JSONArrayPackI64(JSONArrayPack):
    """64-bit integer JSON array packer."""


def json_array_pack_i64(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 64-bit ints to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackI64(arg).to_expr()


ir.StringValue.json_array_pack_i64 = json_array_pack_i64
ir.JSONValue.json_array_pack_i64 = json_array_pack_i64


class JSONArrayPackF32(JSONArrayPack):
    """32-bit float JSON array packer."""


def json_array_pack_f32(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 32-bit floats to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackF32(arg).to_expr()


ir.StringValue.json_array_pack_f32 = json_array_pack_f32
ir.JSONValue.json_array_pack_f32 = json_array_pack_f32


class JSONArrayPackF64(JSONArrayPack):
    """64-bit float JSON array packer."""


def json_array_pack_f64(arg: ir.StringValue | ir.JSONValue) -> ir.BinaryValue:
    """
    Convert a JSON array of zero or more 64-bit floats to an encoded blob.

    Parameters
    ----------
    arg : string
        JSON array

    Returns
    -------
    Blob column

    """
    return JSONArrayPackF64(arg).to_expr()


ir.StringValue.json_array_pack_f64 = json_array_pack_f64
ir.JSONValue.json_array_pack_f64 = json_array_pack_f64
