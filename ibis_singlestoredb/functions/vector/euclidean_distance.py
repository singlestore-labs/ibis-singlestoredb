from __future__ import annotations

import json
from typing import Iterable

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class EuclideanDistance(Value):
    """32-bit float Euclidean distance."""

    left = binary
    right = binary

    output_dtype = dt.double
    output_shape = rlz.shape_like('left')


def euclidean_distance(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 32-bit float input vectors.

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
    return EuclideanDistance(left, right).to_expr()


ir.BinaryValue.euclidean_distance = euclidean_distance


class EuclideanDistanceI8(EuclideanDistance):
    """8-bit integer Euclidean distance."""


def euclidean_distance_i8(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 8-bit int input vectors.

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
    return EuclideanDistanceI8(left, right).to_expr()


ir.BinaryValue.euclidean_distance_i8 = euclidean_distance_i8


class EuclideanDistanceI16(EuclideanDistance):
    """16-bit integer Euclidean distance."""


def euclidean_distance_i16(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 16-bit int input vectors.

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
    return EuclideanDistanceI16(left, right).to_expr()


ir.BinaryValue.euclidean_distance_i16 = euclidean_distance_i16


class EuclideanDistanceI32(EuclideanDistance):
    """32-bit integer Euclidean distance."""


def euclidean_distance_i32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 32-bit int input vectors.

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
    return EuclideanDistanceI32(left, right).to_expr()


ir.BinaryValue.euclidean_distance_i32 = euclidean_distance_i32


class EuclideanDistanceI64(EuclideanDistance):
    """64-bit integer Euclidean distance."""


def euclidean_distance_i64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 64-bit int input vectors.

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
    return EuclideanDistanceI64(left, right).to_expr()


ir.BinaryValue.euclidean_distance_i64 = euclidean_distance_i64


class EuclideanDistanceF32(EuclideanDistance):
    """32-bit float Euclidean distance."""


def euclidean_distance_f32(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 32-bit float input vectors.

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
    return EuclideanDistanceF32(left, right).to_expr()


ir.BinaryValue.euclidean_distance_f32 = euclidean_distance_f32


class EuclideanDistanceF64(EuclideanDistance):
    """64-bit float Euclidean distance."""


def euclidean_distance_f64(
    left: ir.BinaryValue,
    right: ir.BinaryValue | Iterable[float | int],
) -> ir.FloatingValue:
    """
    Return the scalar Euclidean distance between two 64-bit float input vectors.

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
    return EuclideanDistanceF64(left, right).to_expr()


ir.BinaryValue.euclidean_distance_f64 = euclidean_distance_f64
