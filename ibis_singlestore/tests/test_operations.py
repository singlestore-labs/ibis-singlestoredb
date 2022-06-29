import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
ibis.options.verbose = False

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
####### Testing Abs #######
    ("abs()",    "negative_int_c",   lambda x : x.abs(), "SELECT abs(negative_int_c) from test_table"),
    ("abs()",    "int_c",            lambda x : x.abs(), "SELECT abs(int_c) from test_table"),
    ("abs()",    "float_c",          lambda x : x.abs(), "SELECT abs(float_c) from test_table"),
    ("abs()",    "negative_float_c", lambda x : x.abs(), "SELECT abs(negative_float_c) from test_table"),
    # ("abs()",    "bool_c",           lambda x : x.abs(), "SELECT abs(bool_c) from test_table"),
    ("abs()",    "tinyint_c",        lambda x : x.abs(), "SELECT abs(tinyint_c) from test_table"),
    ("abs()",    "smallint_c",       lambda x : x.abs(), "SELECT abs(smallint_c) from test_table"),
    ("abs()",    "mediumint_c",      lambda x : x.abs(), "SELECT abs(mediumint_c) from test_table"),
    ("abs()",    "bigint_c",         lambda x : x.abs(), "SELECT abs(bigint_c) from test_table"),
    ("abs()",    "double_c",         lambda x : x.abs(), "SELECT abs(double_c) from test_table"),
    ("abs()",    "decimal_c",        lambda x : x.abs(), "SELECT abs(decimal_c) from test_table"),
    ("abs()",    "decimal5_c",       lambda x : x.abs(), "SELECT abs(decimal5_c) from test_table"),
    ("abs()",    "numeric_c",        lambda x : x.abs(), "SELECT abs(numeric_c) from test_table"),
    ####### Abs should produce an error with the following types: #######
    ("error",    "date_c",           lambda x : x.abs(), "SELECT abs(date_c) from test_table"),
    ("error",    "time_c",           lambda x : x.abs(), "SELECT abs(time_c) from test_table"),
    ("error",    "time6_c",          lambda x : x.abs(), "SELECT abs(time6_c) from test_table"),
    ("error",    "datetime_c",       lambda x : x.abs(), "SELECT abs(datetime_c) from test_table"),
    ("error",    "datetime6_c",      lambda x : x.abs(), "SELECT abs(datetime6_c) from test_table"),
    ("error",    "timestamp_c",      lambda x : x.abs(), "SELECT abs(timestamp_c) from test_table"),
    ("error",    "timestamp6_c",     lambda x : x.abs(), "SELECT abs(timestamp6_c) from test_table"),
    ("error",    "char32_c",         lambda x : x.abs(), "SELECT abs(char32_c) from test_table"),
    ("error",    "varchar42_c",      lambda x : x.abs(), "SELECT abs(varchar42_c) from test_table"),
    ("error",    "longtext_c",       lambda x : x.abs(), "SELECT abs(longtext_c) from test_table"),
    ("error",    "mediumtext_c",     lambda x : x.abs(), "SELECT abs(mediumtext_c) from test_table"),
    ("error",    "tinytext_c",       lambda x : x.abs(), "SELECT abs(tinytext_c) from test_table"),
    ("error",    "text_c",           lambda x : x.abs(), "SELECT abs(text_c) from test_table"),
    ("error",    "text4_c",          lambda x : x.abs(), "SELECT abs(text4_c) from test_table"),
    ("error",    "blob_c",           lambda x : x.abs(), "SELECT abs(blob_c) from test_table"),
    ("error",    "enumsml_c",        lambda x : x.abs(), "SELECT abs(enumsml_c) from test_table"),
    ("error",    "set_abcd_c",       lambda x : x.abs(), "SELECT abs(set_abcd_c) from test_table"),
####### Testing Add #######
    ("add()",    "negative_int_c",   lambda x : x.add(-3), "SELECT negative_int_c - 3 from test_table"),
    ("add()",    "int_c",            lambda x : x.add(-3), "SELECT int_c - 3 from test_table"),
    ("add()",    "float_c",          lambda x : x.add(-3), "SELECT float_c - 3 from test_table"),
    ("add()",    "negative_float_c", lambda x : x.add(-3), "SELECT negative_float_c - 3 from test_table"),
    # ("add()",    "bool_c",           lambda x : x.add(-3), "SELECT bool_c - 3 from test_table"),
    ("add()",    "tinyint_c",        lambda x : x.add(-3), "SELECT tinyint_c - 3 from test_table"),
    ("add()",    "smallint_c",       lambda x : x.add(-3), "SELECT smallint_c - 3 from test_table"),
    ("add()",    "mediumint_c",      lambda x : x.add(3), "SELECT mediumint_c + 3 from test_table"),
    ("add()",    "bigint_c",         lambda x : x.add(3), "SELECT bigint_c + 3 from test_table"),
    ("add()",    "double_c",         lambda x : x.add(3), "SELECT double_c + 3 from test_table"),
    ("add()",    "decimal_c",        lambda x : x.add(3), "SELECT decimal_c + 3 from test_table"),
    ("add()",    "decimal5_c",       lambda x : x.add(-3), "SELECT decimal5_c - 3 from test_table"),
    ("add()",    "numeric_c",        lambda x : x.add(-3), "SELECT numeric_c - 3 from test_table"),
    ####### Add should produce an error with the following types: #######
    ("error",    "date_c",           lambda x : x.add(-3), "SELECT date_c - 3 from test_table"),
    ("error",    "time_c",           lambda x : x.add(-3), "SELECT time_c - 3 from test_table"),
    ("error",    "time6_c",          lambda x : x.add(-3), "SELECT time6_c - 3 from test_table"),
    ("error",    "datetime_c",       lambda x : x.add(-3), "SELECT datetime_c - 3 from test_table"),
    ("error",    "datetime6_c",      lambda x : x.add(-3), "SELECT datetime6_c - 3 from test_table"),
    ("error",    "timestamp_c",      lambda x : x.add(-3), "SELECT timestamp_c - 3 from test_table"),
    ("error",    "timestamp6_c",     lambda x : x.add(-3), "SELECT timestamp6_c - 3 from test_table"),
    ("error",    "char32_c",         lambda x : x.add(-3), "SELECT char32_c - 3 from test_table"),
    ("error",    "varchar42_c",      lambda x : x.add(-3), "SELECT varchar42_c - 3 from test_table"),
    ("error",    "longtext_c",       lambda x : x.add(-3), "SELECT longtext_c - 3 from test_table"),
    ("error",    "mediumtext_c",     lambda x : x.add(-3), "SELECT mediumtext_c - 3 from test_table"),
    ("error",    "tinytext_c",       lambda x : x.add(-3), "SELECT tinytext_c - 3 from test_table"),
    ("error",    "text_c",           lambda x : x.add(3), "SELECT text_c + 3 from test_table"),
    ("error",    "text4_c",          lambda x : x.add(3), "SELECT text4_c + 3 from test_table"),
    ("error",    "blob_c",           lambda x : x.add(3), "SELECT blob_c + 3 from test_table"),
    ("error",    "enumsml_c",        lambda x : x.add(-3), "SELECT enumsml_c - 3 from test_table"),
    ("error",    "set_abcd_c",       lambda x : x.add(-3), "SELECT set_abcd_c - 3 from test_table"),
####### Testing Capitalize #######
    ("capitalize()",    "char32_c",         lambda x : x.capitalize(), "SELECT UPPER(char32_c) from test_table"),
    ("capitalize()",    "varchar42_c",      lambda x : x.capitalize(), "SELECT UPPER(varchar42_c) from test_table"),
    ("capitalize()",    "longtext_c",       lambda x : x.capitalize(), "SELECT UPPER(longtext_c) from test_table"),
    ("capitalize()",    "mediumtext_c",     lambda x : x.capitalize(), "SELECT UPPER(mediumtext_c) from test_table"),
    ("capitalize()",    "tinytext_c",       lambda x : x.capitalize(), "SELECT UPPER(tinytext_c) from test_table"),
    ("capitalize()",    "text_c",           lambda x : x.capitalize(), "SELECT UPPER(text_c) from test_table"),
    ("capitalize()",    "text4_c",          lambda x : x.capitalize(), "SELECT UPPER(text4_c) from test_table"),
    ("error",    "negative_int_c",   lambda x : x.capitalize(), "SELECT UPPER(negative_int_c) from test_table"),
    ("capitalize()",    "set_abcd_c",       lambda x : x.capitalize(), "SELECT UPPER(set_abcd_c) from test_table"),
    ("error",    "int_c",            lambda x : x.capitalize(), "SELECT UPPER(int_c) from test_table"),
    ("error",    "float_c",          lambda x : x.capitalize(), "SELECT UPPER(float_c) from test_table"),
    ("error",    "negative_float_c", lambda x : x.capitalize(), "SELECT UPPER(negative_float_c) from test_table"),
    # ("error",    "bool_c",           lambda x : x.capitalize(), "SELECT UPPER(bool_c) from test_table"),
    ("error",    "tinyint_c",        lambda x : x.capitalize(), "SELECT UPPER(tinyint_c) from test_table"),
    ("error",    "smallint_c",       lambda x : x.capitalize(), "SELECT UPPER(smallint_c) from test_table"),
    ("error",    "mediumint_c",      lambda x : x.capitalize(), "SELECT UPPER(mediumint_c) from test_table"),
    ("error",    "bigint_c",         lambda x : x.capitalize(), "SELECT UPPER(bigint_c) from test_table"),
    ("error",    "double_c",         lambda x : x.capitalize(), "SELECT UPPER(double_c) from test_table"),
    ("error",    "decimal_c",        lambda x : x.capitalize(), "SELECT UPPER(decimal_c) from test_table"),
    ("error",    "decimal5_c",       lambda x : x.capitalize(), "SELECT UPPER(decimal5_c) from test_table"),
    ("error",    "numeric_c",        lambda x : x.capitalize(), "SELECT UPPER(numeric_c) from test_table"),
    ####### Capitalize should produce an error with the following types: #######
    ("error",    "date_c",           lambda x : x.capitalize(), "SELECT UPPER(date_c) from test_table"),
    ("error",    "time_c",           lambda x : x.capitalize(), "SELECT UPPER(time_c) from test_table"),
    ("error",    "time6_c",          lambda x : x.capitalize(), "SELECT UPPER(time6_c) from test_table"),
    ("error",    "datetime_c",       lambda x : x.capitalize(), "SELECT UPPER(datetime_c) from test_table"),
    ("error",    "datetime6_c",      lambda x : x.capitalize(), "SELECT UPPER(datetime6_c) from test_table"),
    ("error",    "timestamp_c",      lambda x : x.capitalize(), "SELECT UPPER(timestamp_c) from test_table"),
    ("error",    "timestamp6_c",     lambda x : x.capitalize(), "SELECT UPPER(timestamp6_c) from test_table"),
    ("error",    "blob_c",           lambda x : x.capitalize(), "SELECT UPPER(blob_c) from test_table"),
    ("error",    "enumsml_c",        lambda x : x.capitalize(), "SELECT UPPER(enumsml_c) from test_table"),
####### Testing Cast #######
    ("cast()",    "negative_int_c",   lambda x : x.cast(dt.string), "SELECT cast(negative_int_c AS char) from test_table"),
    ("cast()",    "int_c",            lambda x : x.cast(dt.string), "SELECT cast(int_c AS char) from test_table"),
    ("cast()",    "float_c",          lambda x : x.cast(dt.string), "SELECT cast(float_c AS char) from test_table"),
    ("cast()",    "negative_float_c", lambda x : x.cast(dt.string), "SELECT cast(negative_float_c AS char) from test_table"),
    # ("cast()",    "bool_c",           lambda x : x.cast(dt.string), "SELECT cast(bool_c AS char) from test_table"),
    ("cast()",    "tinyint_c",        lambda x : x.cast(dt.string), "SELECT cast(tinyint_c AS char) from test_table"),
    ("cast()",    "smallint_c",       lambda x : x.cast(dt.string), "SELECT cast(smallint_c AS char) from test_table"),
    ("cast()",    "mediumint_c",      lambda x : x.cast(dt.string), "SELECT cast(mediumint_c AS char) from test_table"),
    ("cast()",    "bigint_c",         lambda x : x.cast(dt.string), "SELECT cast(bigint_c AS char) from test_table"),
    ("cast()",    "double_c",         lambda x : x.cast(dt.string), "SELECT cast(double_c AS char) from test_table"),
    ("cast()",    "decimal_c",        lambda x : x.cast(dt.string), "SELECT cast(decimal_c AS char) from test_table"),
    ("cast()",    "decimal5_c",       lambda x : x.cast(dt.string), "SELECT cast(decimal5_c AS char) from test_table"),
    ("cast()",    "numeric_c",        lambda x : x.cast(dt.string), "SELECT cast(numeric_c AS char) from test_table"),
    ####### Cast should produce an error with the following types: #######
    ("cast()",    "date_c",           lambda x : x.cast(dt.string), "SELECT cast(date_c AS char) from test_table"),
    ("cast()",    "time_c",           lambda x : x.cast(dt.string), "SELECT cast(time_c AS char) from test_table"),
    ("cast()",    "time6_c",          lambda x : x.cast(dt.string), "SELECT cast(time6_c AS char) from test_table"),
    ("cast()",    "datetime_c",       lambda x : x.cast(dt.string), "SELECT cast(datetime_c AS char) from test_table"),
    ("cast()",    "datetime6_c",      lambda x : x.cast(dt.string), "SELECT cast(datetime6_c AS char) from test_table"),
    ("cast()",    "timestamp_c",      lambda x : x.cast(dt.string), "SELECT cast(timestamp_c AS char) from test_table"),
    ("cast()",    "timestamp6_c",     lambda x : x.cast(dt.string), "SELECT cast(timestamp6_c AS char) from test_table"),
    ("cast()",    "char32_c",         lambda x : x.cast(dt.string), "SELECT cast(char32_c AS char) from test_table"),
    ("cast()",    "varchar42_c",      lambda x : x.cast(dt.string), "SELECT cast(varchar42_c AS char) from test_table"),
    ("cast()",    "longtext_c",       lambda x : x.cast(dt.string), "SELECT cast(longtext_c AS char) from test_table"),
    ("cast()",    "mediumtext_c",     lambda x : x.cast(dt.string), "SELECT cast(mediumtext_c AS char) from test_table"),
    ("cast()",    "tinytext_c",       lambda x : x.cast(dt.string), "SELECT cast(tinytext_c AS char) from test_table"),
    ("cast()",    "text_c",           lambda x : x.cast(dt.string), "SELECT cast(text_c AS char) from test_table"),
    ("cast()",    "text4_c",          lambda x : x.cast(dt.string), "SELECT cast(text4_c AS char) from test_table"),
    ("cast()",    "blob_c",           lambda x : x.cast(dt.string), "SELECT cast(blob_c AS char) from test_table"),
    ("error",    "enumsml_c",        lambda x : x.cast(dt.string), "SELECT cast(enumsml_c AS char) from test_table"),
    ("cast()",    "set_abcd_c",       lambda x : x.cast(dt.string), "SELECT cast(set_abcd_c AS char) from test_table"),
####### Testing Ceil #######
    ("ceil()",    "negative_int_c",   lambda x : x.ceil(), "SELECT ceil(negative_int_c) from test_table"),
    ("ceil()",    "int_c",            lambda x : x.ceil(), "SELECT ceil(int_c) from test_table"),
    ("ceil()",    "float_c",          lambda x : x.ceil(), "SELECT ceil(float_c) from test_table"),
    ("ceil()",    "negative_float_c", lambda x : x.ceil(), "SELECT ceil(negative_float_c) from test_table"),
    # ("ceil()",    "bool_c",           lambda x : x.ceil(), "SELECT ceil(bool_c) from test_table"),
    ("ceil()",    "tinyint_c",        lambda x : x.ceil(), "SELECT ceil(tinyint_c) from test_table"),
    ("ceil()",    "smallint_c",       lambda x : x.ceil(), "SELECT ceil(smallint_c) from test_table"),
    ("ceil()",    "mediumint_c",      lambda x : x.ceil(), "SELECT ceil(mediumint_c) from test_table"),
    ("ceil()",    "bigint_c",         lambda x : x.ceil(), "SELECT ceil(bigint_c) from test_table"),
    ("ceil()",    "double_c",         lambda x : x.ceil(), "SELECT ceil(double_c) from test_table"),
    ("ceil()",    "decimal_c",        lambda x : x.ceil(), "SELECT ceil(decimal_c) from test_table"),
    ("ceil()",    "decimal5_c",       lambda x : x.ceil(), "SELECT ceil(decimal5_c) from test_table"),
    ("ceil()",    "numeric_c",        lambda x : x.ceil(), "SELECT ceil(numeric_c) from test_table"),
    ####### Ceil should produce an error with the following types: #######
    ("error",    "date_c",           lambda x : x.ceil(), "SELECT ceil(date_c) from test_table"),
    ("error",    "time_c",           lambda x : x.ceil(), "SELECT ceil(time_c) from test_table"),
    ("error",    "time6_c",          lambda x : x.ceil(), "SELECT ceil(time6_c) from test_table"),
    ("error",    "datetime_c",       lambda x : x.ceil(), "SELECT ceil(datetime_c) from test_table"),
    ("error",    "datetime6_c",      lambda x : x.ceil(), "SELECT ceil(datetime6_c) from test_table"),
    ("error",    "timestamp_c",      lambda x : x.ceil(), "SELECT ceil(timestamp_c) from test_table"),
    ("error",    "timestamp6_c",     lambda x : x.ceil(), "SELECT ceil(timestamp6_c) from test_table"),
    ("error",    "char32_c",         lambda x : x.ceil(), "SELECT ceil(char32_c) from test_table"),
    ("error",    "varchar42_c",      lambda x : x.ceil(), "SELECT ceil(varchar42_c) from test_table"),
    ("error",    "longtext_c",       lambda x : x.ceil(), "SELECT ceil(longtext_c) from test_table"),
    ("error",    "mediumtext_c",     lambda x : x.ceil(), "SELECT ceil(mediumtext_c) from test_table"),
    ("error",    "tinytext_c",       lambda x : x.ceil(), "SELECT ceil(tinytext_c) from test_table"),
    ("error",    "text_c",           lambda x : x.ceil(), "SELECT ceil(text_c) from test_table"),
    ("error",    "text4_c",          lambda x : x.ceil(), "SELECT ceil(text4_c) from test_table"),
    ("error",    "blob_c",           lambda x : x.ceil(), "SELECT ceil(blob_c) from test_table"),
    ("error",    "enumsml_c",        lambda x : x.ceil(), "SELECT ceil(enumsml_c) from test_table"),
    ("error",    "set_abcd_c",       lambda x : x.ceil(), "SELECT ceil(set_abcd_c) from test_table")
]

@pytest.mark.parametrize(
    ("operation", "column", "ibis_operation", "expected_operation"),
    [
        param(operation, column, ibis_operation, expected_operation, id=operation)
        for   operation, column, ibis_operation, expected_operation  in OPERATIONS
    ],
)

def test_operations(con, operation, column, ibis_operation, expected_operation):
    test_table = con.table('test_table')
    if operation == "error":
        try:
            ibis_result = ibis_operation(getattr(test_table, column)).name(column).execute()
            expected_result = pd.DataFrame(con.raw_sql(expected_operation)).squeeze()
        except AttributeError:
            return
        assert False
    
    ibis_result = ibis_operation(getattr(test_table, column)).name(column).execute().sort_values()
    expected_result = pd.DataFrame(con.raw_sql(expected_operation)).squeeze().sort_values()
    ibis_result.equals(expected_result)