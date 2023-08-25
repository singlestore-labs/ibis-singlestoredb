from __future__ import annotations

from typing import Optional

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations.core import Value

binary = rlz.value(dt.binary)


class VectorSort(Value):
    """32-bit float vector sort."""

    arg = binary
    direction = rlz.optional(rlz.string)

    output_dtype = dt.binary
    output_shape = rlz.shape_like('arg')


def vector_sort(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSort(arg, direction).to_expr()


ir.BinaryValue.vector_sort = vector_sort


class VectorSortI8(VectorSort):
    """8-bit integer vector sort."""


def vector_sort_i8(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in an 8-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortI8(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i8 = vector_sort_i8


class VectorSortI16(VectorSort):
    """16-bit integer vector sort."""


def vector_sort_i16(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 16-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortI16(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i16 = vector_sort_i16


class VectorSortI32(VectorSort):
    """32-bit integer vector sort."""


def vector_sort_i32(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortI32(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i32 = vector_sort_i32


class VectorSortI64(VectorSort):
    """64-bit integer vector sort."""


def vector_sort_i64(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 64-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortI64(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i64 = vector_sort_i64


class VectorSortF32(VectorSort):
    """32-bit float vector sort."""


def vector_sort_f32(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortF32(arg, direction).to_expr()


ir.BinaryValue.vector_sort_f32 = vector_sort_f32


class VectorSortF64(VectorSort):
    """64-bit float vector sort."""


def vector_sort_f64(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc',
) -> ir.BinaryValue:
    """
    Sort the elements in a 64-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    if direction is ibis.desc:
        direction = 'desc'
    elif direction is ibis.asc:
        direction = 'asc'
    return VectorSortF64(arg, direction).to_expr()


ir.BinaryValue.vector_sort_f64 = vector_sort_f64
