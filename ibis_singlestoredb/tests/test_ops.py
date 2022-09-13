#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import math

import ibis
import ibis.expr.datatypes as dt
import pytest
from pytest import param

ibis.options.verbose = False


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError('decimal places must be an integer.')
    elif decimals < 0:
        raise ValueError('decimal places has to be 0 or more.')
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


#
# OPERATIONS TESTED
#
# Abs
# Add
# Capitalize
# Cast
# Ceil

#
# OPERATIONS REMAINING
#
# Alias
# And
# BitAnd
# BitOr
# BitXor
# Bucket
# CategoryLabel
# Coalesce
# CountDistinct
# Count
# CumulativeMax
# CumulativeMean
# CumulativeMin
# CumulativeSum
# DateAdd
# DateDiff
# DateFromYMD
# DateSub
# DateTruncate
# Date
# DayOfWeekIndex
# DayOfWeekName
# DenseRank
# EndsWith
# Equals
# ExistsSubquery
# Exp
# ExtractDayOfYear
# ExtractDay
# ExtractEpochSeconds
# ExtractHour
# ExtractMillisecond
# ExtractMinute
# ExtractMonth
# ExtractQuarter
# ExtractSecond
# ExtractWeekOfYear
# ExtractYear
# FirstValue
# FloorDivide
# Floor
# GeoArea
# GeoAsBinary
# GeoAsEWKB
# GeoAsEWKT
# GeoAsText
# GeoAzimuth
# GeoBuffer
# GeoCentroid
# GeoContainsProperly
# GeoContains
# GeoCoveredBy
# GeoCovers
# GeoCrosses
# GeoDFullyWithin
# GeoDWithin
# GeoDifference
# GeoDisjoint
# GeoDistance
# GeoEndPoint
# GeoEnvelope
# GeoEquals
# GeoGeometryN
# GeoGeometryType
# GeoIntersection
# GeoIntersects
# GeoIsValid
# GeoLength
# GeoLineLocatePoint
# GeoLineMerge
# GeoLineSubstring
# GeoNPoints
# GeoOrderingEquals
# GeoOverlaps
# GeoPerimeter
# GeoSRID
# GeoSetSRID
# GeoSimplify
# GeoStartPoint
# GeoTouches
# GeoTransform
# GeoUnaryUnion
# GeoUnion
# GeoWithin
# GeoX
# GeoY
# GreaterEqual
# Greater
# Greatest
# GroupConcat
# IdenticalTo
# IfNull
# IntervalFromInteger
# IsNull
# LPad
# LStrip
# Lag
# LastValue
# Lead
# Least
# LessEqual
# Less
# Ln
# Log10
# Log2
# Log
# Lowercase
# Max
# Mean
# MinRank
# Min
# Modulus
# Multiply
# NTile
# Negate
# NotAll
# NotAny
# NotContains
# NotEquals
# NotNull
# Not
# NullIfZero
# NullIf
# NullLiteral
# Or
# PercentRank
# Power
# RPad
# RStrip
# RandomScalar
# RegexSearch
# Repeat
# Reverse
# Round
# RowNumber
# SearchedCase
# Sign
# SimpleCase
# Sqrt
# StandardDev
# StartsWith
# StrRight
# Strftime
# StringAscii
# StringConcat
# StringContains
# StringFind
# StringJoin
# StringLength
# StringReplace
# StringSQLILike
# Strip
# Substring
# Subtract
# Sum
# TableArrayView
# TableColumn
# TimeFromHMS
# TimestampAdd
# TimestampDiff
# TimestampFromYMDHMS
# TimestampNow
# TimestampSub
# TimestampTruncate
# TypeOf
# Uppercase
# ValueList
# Variance
# WindowOp

OPERATIONS = [
    # Testing Abs
    ('abs()', 'negative_int_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'int_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'float_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'negative_float_c', lambda x: x.abs(), lambda x: abs(x)),
    # ("abs()", "bool_c", lambda x : x.abs(), lambda x : abs(x)),
    ('abs()', 'tinyint_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'smallint_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'mediumint_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'bigint_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'double_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'decimal_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'decimal5_c', lambda x: x.abs(), lambda x: abs(x)),
    ('abs()', 'numeric_c', lambda x: x.abs(), lambda x: abs(x)),

    # Abs should produce an error with the following types:
    ('error', 'date_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'time_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'time6_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'datetime_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'datetime6_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'timestamp_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'timestamp6_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'char32_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'varchar42_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'longtext_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'mediumtext_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'tinytext_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'text_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'text4_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'blob_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'enumsml_c', lambda x: x.abs(), lambda x: abs(x)),
    ('error', 'set_abcd_c', lambda x: x.abs(), lambda x: abs(x)),

    # Testing Add
    ('add()', 'int_c', lambda x: x.add(3), lambda x: x + 3),
    ('add()', 'int_c', lambda x: x.add(-5), lambda x: x - 5),
    ('add()', 'negative_int_c', lambda x: x.add(3), lambda x: x + 3),
    ('add()', 'negative_int_c', lambda x: x.add(-5), lambda x: x - 5),
    ('add()', 'float_c', lambda x: x.add(3), lambda x: x + 3),
    ('add()', 'float_c', lambda x: x.add(-5), lambda x: x - 5),
    ('add()', 'float_c', lambda x: x.add(-5.3), lambda x: x - 5.3),
    # ("add()", "bool_c", lambda x : x.add(-3), lambda x : x - 3),
    ('add()', 'tinyint_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'smallint_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'mediumint_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'bigint_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'double_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'decimal_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'decimal5_c', lambda x: x.add(-3), lambda x: x - 3),
    ('add()', 'numeric_c', lambda x: x.add(-3), lambda x: x - 3),

    # Add should produce an error with the following types:
    ('error', 'date_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'time_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'time6_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'datetime_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'datetime6_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'timestamp_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'timestamp6_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'char32_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'varchar42_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'longtext_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'mediumtext_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'tinytext_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'text_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'text4_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'blob_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'enumsml_c', lambda x: x.add(-3), lambda x: x - 3),
    ('error', 'set_abcd_c', lambda x: x.add(-3), lambda x: x - 3),

    # Testing And
    # ("And()", "bool_c", lambda x: x.And(True), lambda x: x and True),
    ('error', 'int_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'int_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'negative_int_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'negative_int_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'float_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'float_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'float_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'tinyint_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'smallint_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'mediumint_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'bigint_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'double_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'decimal_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'decimal5_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'numeric_c', lambda x: x.And(True), lambda x: x and True),

    # And should produce an error with the following types:
    ('error', 'date_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'time_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'time6_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'datetime_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'datetime6_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'timestamp_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'timestamp6_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'char32_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'varchar42_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'longtext_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'mediumtext_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'tinytext_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'text_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'text4_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'blob_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'enumsml_c', lambda x: x.And(True), lambda x: x and True),
    ('error', 'set_abcd_c', lambda x: x.And(True), lambda x: x and True),

    # Testing Capitalize
    ('capitalize()', 'char32_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'varchar42_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'longtext_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'mediumtext_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'tinytext_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'text_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('capitalize()', 'text4_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    # ("capitalize()", "set_abcd_c", lambda x: x.capitalize(),
    #                                lambda x: list([i.capitalize() for i in x])),

    # Capitalize should produce an error with the following types:
    ('error', 'negative_int_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'int_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'float_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'negative_float_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    # ("error", "bool_c", lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'tinyint_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'smallint_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'mediumint_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'bigint_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'double_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'decimal_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'decimal5_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'numeric_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'date_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'time_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'time6_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'datetime_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'datetime6_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'timestamp_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'timestamp6_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'blob_c', lambda x: x.capitalize(), lambda x: x.capitalize()),
    ('error', 'enumsml_c', lambda x: x.capitalize(), lambda x: x.capitalize()),

    # Testing Cast
    ('cast()', 'negative_int_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'int_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'float_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'negative_float_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "bool_c", lambda x : x.cast(dt.string), lambda x : str(x)),
    ('cast()', 'tinyint_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'smallint_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'mediumint_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'bigint_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "double_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "decimal_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "decimal5_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "numeric_c", lambda x: x.cast(dt.string), lambda x: str(x)),

    # Cast should produce an error with the following types:
    # ("cast()", "date_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "time_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "time6_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'datetime_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "datetime6_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'timestamp_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "timestamp6_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'char32_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'varchar42_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'longtext_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'mediumtext_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'tinytext_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'text_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    ('cast()', 'text4_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "blob_c", lambda x: x.cast(dt.string), lambda x: str(x)),
    ('error',  'enumsml_c', lambda x: x.cast(dt.string), lambda x: str(x)),
    # ("cast()", "set_abcd_c", lambda x: x.cast(dt.string), lambda x: str(x)),

    # Testing Ceil
    ('ceil()', 'negative_int_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'int_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'float_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'negative_float_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    # ("ceil()", "bool_c", lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'tinyint_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'smallint_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'mediumint_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'bigint_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'double_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'decimal_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'decimal5_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('ceil()', 'numeric_c', lambda x: x.ceil(), lambda x: math.ceil(x)),

    # Ceil should produce an error with the following types:
    ('error', 'date_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'time_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'time6_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'datetime_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'datetime6_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'timestamp_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'timestamp6_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'char32_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'varchar42_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'longtext_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'mediumtext_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'tinytext_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'text_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'text4_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'blob_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'enumsml_c', lambda x: x.ceil(), lambda x: math.ceil(x)),
    ('error', 'set_abcd_c', lambda x: x.ceil(), lambda x: math.ceil(x)),

    ('log(10)', 'int_c', lambda x: x.log(10), lambda x: math.log(x, 10)),
    ('log(2)', 'int_c', lambda x: x.log(2), lambda x: math.log(x, 2)),
    ('log10()', 'int_c', lambda x: x.log10(), lambda x: math.log10(x)),
    ('round()', 'float_c', lambda x: x.round(), lambda x: round(x)),
]


@pytest.mark.parametrize(
    ('test_name', 'column', 'ibis_operation', 'expected_operation'),
    [
        param(test_name, column, ibis_operation, expected_operation, id=test_name)
        for test_name, column, ibis_operation, expected_operation in OPERATIONS
    ],
)
def test_operations(con, test_name, column, ibis_operation, expected_operation):
    datatypes = con.table('datatypes')
    if test_name == 'error':
        try:
            out = datatypes[
                getattr(datatypes, column),
                ibis_operation(getattr(datatypes, column)).name('_'),
            ].execute()
            original = list(out.iloc[:, 0])
            result = list(out.iloc[:, 1])
        except AttributeError:
            return
        assert False

    out = datatypes[
        getattr(datatypes, column),
        ibis_operation(getattr(datatypes, column)).name('_'),
    ].execute()
    original = list(out.iloc[:, 0])
    result = list(out.iloc[:, 1])

    for i, value in enumerate(original):
        ibis_result = result[i]
        expected_result = expected_operation(value)

        if test_name == 'abs()' and type(ibis_result) == float:
            ibis_result = truncate(ibis_result, 2)
            expected_result = truncate(expected_result, 2)

        if test_name == 'add()' and type(ibis_result) == float:
            if column == 'decimal5_c':
                ibis_result = round(ibis_result, 3)
                expected_result = round(expected_result, 3)
            else:
                ibis_result = truncate(ibis_result, 3)
                expected_result = truncate(expected_result, 3)

        assert ibis_result == expected_result
