from __future__ import annotations

import json
from typing import Iterable

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorMul(Value):
    """32-bit float vector multiplication."""

    left = binary
    right = binary

    output_dtype = dt.binary
    output_shape = rlz.shape_like('left')


def vector_mul(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 32-bit float vector blobs.

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
    return VectorMul(left, right).to_expr()


ir.BinaryValue.vector_mul = vector_mul


class VectorMulI8(VectorMul):
    """8-bit integer vector multiplication."""


def vector_mul_i8(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 8-bit int vector blobs.

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
    return VectorMulI8(left, right).to_expr()


ir.BinaryValue.vector_mul_i8 = vector_mul_i8


class VectorMulI16(VectorMul):
    """16-bit integer vector multiplication."""


def vector_mul_i16(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 16-bit int vector blobs.

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
    return VectorMulI16(left, right).to_expr()


ir.BinaryValue.vector_mul_i16 = vector_mul_i16


class VectorMulI32(VectorMul):
    """32-bit integer vector multiplication."""


def vector_mul_i32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 32-bit int vector blobs.

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
    return VectorMulI32(left, right).to_expr()


ir.BinaryValue.vector_mul_i32 = vector_mul_i32


class VectorMulI64(VectorMul):
    """64-bit integer vector multiplication."""


def vector_mul_i64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 64-bit int vector blobs.

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
    return VectorMulI64(left, right).to_expr()


ir.BinaryValue.vector_mul_i64 = vector_mul_i64


class VectorMulF32(VectorMul):
    """32-bit float vector multiplication."""


def vector_mul_f32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 32-bit float vector blobs.

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
    return VectorMulF32(left, right).to_expr()


ir.BinaryValue.vector_mul_f32 = vector_mul_f32


class VectorMulF64(VectorMul):
    """64-bit float vector multiplication."""


def vector_mul_f64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.BinaryValue:
    """
    Multiplies two 64-bit float vector blobs.

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
    return VectorMulF64(left, right).to_expr()


ir.BinaryValue.vector_mul_f64 = vector_mul_f64
