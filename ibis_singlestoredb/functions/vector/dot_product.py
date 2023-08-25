from __future__ import annotations

import json
from typing import Iterable

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class DotProduct(Value):
    """32-bit float dot product."""

    left = binary
    right = binary

    output_dtype = dt.float64
    output_shape = rlz.shape_like('left')


def dot_product(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 32-bit float vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack()
    return DotProduct(left, right).to_expr()


ir.BinaryValue.dot_product = dot_product


class DotProductI8(DotProduct):
    """8-bit integer dot product."""


def dot_product_i8(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 8-bit int vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i8()
    return DotProductI8(left, right).to_expr()


ir.BinaryValue.dot_product_i8 = dot_product_i8


class DotProductI16(DotProduct):
    """16-bit integer dot product."""


def dot_product_i16(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 16-bit int vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i16()
    return DotProductI16(left, right).to_expr()


ir.BinaryValue.dot_product_i16 = dot_product_i16


class DotProductI32(DotProduct):
    """32-bit integer dot product."""


def dot_product_i32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 32-bit int vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i32()
    return DotProductI32(left, right).to_expr()


ir.BinaryValue.dot_product_i32 = dot_product_i32


class DotProductI64(DotProduct):
    """64-bit integer dot product."""


def dot_product_i64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 64-bit int vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_i64()
    return DotProductI64(left, right).to_expr()


ir.BinaryValue.dot_product_i64 = dot_product_i64


class DotProductF32(DotProduct):
    """32-bit float dot product."""


def dot_product_f32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 32-bit float vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_f32()
    return DotProductF32(left, right).to_expr()


ir.BinaryValue.dot_product_f32 = dot_product_f32


class DotProductF64(DotProduct):
    """64-bit float dot product."""


def dot_product_f64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar product, or dot product of two 64-bit float vector input values.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob or iterable
        Vector expression

    Returns
    -------
    Double column

    """
    if not isinstance(right, ir.Value):
        right = ibis.literal(json.dumps(list(right))).json_array_pack_f64()
    return DotProductF64(left, right).to_expr()


ir.BinaryValue.dot_product_f64 = dot_product_f64
