import pytest
from pytest import param

import numpy

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
ibis.options.verbose = True

####### OPERATIONS TESTED #######
# Abs
# Add
# Capitalize
# Cast
# Ceil

####### OPERATIONS REMAINING #######
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
    ("abs()",           lambda x : x.abs(),             "SELECT ABS({}) from test_table"),
    ("add()",           lambda x : x.add(3),            "SELECT {} + 3 from test_table"),
    ("capitalize()",    lambda x : x.capitalize(),      "SELECT ucase({}) from test_table"),
    ("cast()",          lambda x : x.cast(dt.string),   "SELECT CAST({} AS CHAR) from test_table"),
    ("ceil()",          lambda x : x.ceil(),            "SELECT CEIL({}) from test_table"),
    ("Alias()",         lambda x : x.alias(),           "SELECT {} AS alias from test_table"),
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
]

COLUMNS = [
    'tinyint_c',
    'smallint_c',
    'mediumint_c',
    'int_c',
    'bigint_c',
    'float_c',
    'double_c',
    'decimal_c',
    'decimal5_c',
    'numeric_c',
    'date_c',
    'time_c',
    'time6_c',
    'datetime_c',
    'datetime6_c',
    'timestamp_c',
    'timestamp6_c',
    'char32_c',
    'varchar42_c',
    'longtext_c',
    'mediumtext_c',
    'tinytext_c',
    'text_c',
    'text4_c',
    'blob_c',
    'enum_sml_c',
    'set_abcd_c',
    'negative_int_c',
    'negative_float_c',
    'bool_c'
]

@pytest.mark.parametrize(
    ("operation", "column", "ibis_operation", "expected_operation"),
    [
        param(operation, column, ibis_operation, expected_operation.format(column), id=operation)
        for   operation, ibis_operation, expected_operation in OPERATIONS
        for   column in COLUMNS
    ],
)

def test_operations(con, operation, column, ibis_operation, expected_operation):
    test_table = con.table('test_table')

    try:
        ibis_result = ibis_operation(getattr(test_table, column)).name("_").execute().sort_values()
        expected_result = pd.DataFrame(con.raw_sql(expected_operation)).squeeze().sort_values()
    except AttributeError:
        return
    pd.testing.assert_series_equal(ibis_result, expected_result, rtol=1e-05, atol=1e-08,
                                   check_names=False,
                                   check_dtype=False,
                                   check_index=False,
                                   check_index_type=False)