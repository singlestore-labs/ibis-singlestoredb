from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorSubvector(Value):
    """32-bit float subsection."""

    arg = binary
    start = rlz.integer
    length = rlz.integer

    output_dtype = dt.binary
    output_shape = rlz.shape_like('arg')


def vector_subvector(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvector(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector = vector_subvector


class VectorSubvectorI8(VectorSubvector):
    """8-bit integer subsection."""


def vector_subvector_i8(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of an 8-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorI8(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_i8 = vector_subvector_i8


class VectorSubvectorI16(VectorSubvector):
    """16-bit integer subsection."""


def vector_subvector_i16(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 16-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorI16(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_i16 = vector_subvector_i16


class VectorSubvectorI32(VectorSubvector):
    """32-bit integer subsection."""


def vector_subvector_i32(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 32-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorI32(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_i32 = vector_subvector_i32


class VectorSubvectorI64(VectorSubvector):
    """64-bit integer subsection."""


def vector_subvector_i64(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 64-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorI64(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_i64 = vector_subvector_i64


class VectorSubvectorF32(VectorSubvector):
    """32-bit float subsection."""


def vector_subvector_f32(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorF32(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_f32 = vector_subvector_f32


class VectorSubvectorF64(VectorSubvector):
    """64-bit float subsection."""


def vector_subvector_f64(
    arg: ir.BinaryValue, start: ir.IntegerValue, length: ir.IntegerValue,
) -> ir.BinaryValue:
    """
    Derive a subsection of a 64-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    start : int
        Zero-indexed position to start
    length : int
        Length of the subvector

    Returns
    -------
    Blob column

    """
    return VectorSubvectorF64(arg, start, length).to_expr()


ir.BinaryValue.vector_subvector_f64 = vector_subvector_f64
