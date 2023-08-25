from __future__ import annotations

import json
from typing import Iterable

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorAdd(Value):
    """32-bit float vector addition."""

    left = binary
    right = binary

    output_dtype = dt.binary
    output_shape = rlz.shape_like('left')


def vector_add(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 32-bit float input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack()
    return VectorAdd(left, right).to_expr()


ir.BinaryValue.vector_add = vector_add


class VectorAddI8(VectorAdd):
    """8-bit integer vector addition."""


def vector_add_i8(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 8-bit int input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i8()
    return VectorAddI8(left, right).to_expr()


ir.BinaryValue.vector_add_i8 = vector_add_i8


class VectorAddI16(VectorAdd):
    """16-bit integer vector addition."""


def vector_add_i16(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 16-bit int input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i16()
    return VectorAddI16(left, right).to_expr()


ir.BinaryValue.vector_add_i16 = vector_add_i16


class VectorAddI32(VectorAdd):
    """32-bit integer vector addition."""


def vector_add_i32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 32-bit int input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i32()
    return VectorAddI32(left, right).to_expr()


ir.BinaryValue.vector_add_i32 = vector_add_i32


class VectorAddI64(VectorAdd):
    """64-bit integer vector addition."""


def vector_add_i64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 64-bit int input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i64()
    return VectorAddI64(left, right).to_expr()


ir.BinaryValue.vector_add_i64 = vector_add_i64


class VectorAddF32(VectorAdd):
    """32-bit float vector addition."""


def vector_add_f32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 32-bit float input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_f32()
    return VectorAddF32(left, right).to_expr()


ir.BinaryValue.vector_add_f32 = vector_add_f32


class VectorAddF64(VectorAdd):
    """64-bit float vector addition."""


def vector_add_f64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Add two 64-bit float input vectors.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Blob column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_f64()
    return VectorAddF64(left, right).to_expr()


ir.BinaryValue.vector_add_f64 = vector_add_f64
