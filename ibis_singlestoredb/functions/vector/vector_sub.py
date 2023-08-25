from __future__ import annotations

import json
from typing import Iterable

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorSub(Value):
    """32-bit float vector subtraction."""

    left = binary
    right = binary

    output_dtype = dt.binary
    output_shape = rlz.shape_like('left')


def vector_sub(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 32-bit float vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack()
    return VectorSub(left, right).to_expr()


ir.BinaryValue.vector_sub = vector_sub


class VectorSubI8(VectorSub):
    """8-bit integer vector subtraction."""


def vector_sub_i8(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 8-bit int vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_i8()
    return VectorSubI8(left, right).to_expr()


ir.BinaryValue.vector_sub_i8 = vector_sub_i8


class VectorSubI16(VectorSub):
    """16-bit integer vector subtraction."""


def vector_sub_i16(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 16-bit int vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_i16()
    return VectorSubI16(left, right).to_expr()


ir.BinaryValue.vector_sub_i16 = vector_sub_i16


class VectorSubI32(VectorSub):
    """32-bit integer vector subtraction."""


def vector_sub_i32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 32-bit int vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_i32()
    return VectorSubI32(left, right).to_expr()


ir.BinaryValue.vector_sub_i32 = vector_sub_i32


class VectorSubI64(VectorSub):
    """64-bit integer vector subtraction."""


def vector_sub_i64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 64-bit int vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_i64()
    return VectorSubI64(left, right).to_expr()


ir.BinaryValue.vector_sub_i64 = vector_sub_i64


class VectorSubF32(VectorSub):
    """32-bit float vector subtraction."""


def vector_sub_f32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 32-bit float vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_f32()
    return VectorSubF32(left, right).to_expr()


ir.BinaryValue.vector_sub_f32 = vector_sub_f32


class VectorSubF64(VectorSub):
    """64-bit float vector subtraction."""


def vector_sub_f64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Subtract the second 64-bit float vector from the first.

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
        right = ibis.literal(json.dumps(right)).json_array_pack_f64()
    return VectorSubF64(left, right).to_expr()


ir.BinaryValue.vector_sub_f64 = vector_sub_f64
