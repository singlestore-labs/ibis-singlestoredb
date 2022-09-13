#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import pytest
from pytest import param

ibis.options.verbose = False

OPERATIONS = [
    # Abs
    ('abs()', lambda x: x.abs()),
    # Add
    ('add()', lambda x: x.add(3)),
    # Capitalize
    ('capitalize()', lambda x: x.capitalize()),
    # Case
    ('cast()', lambda x: x.cast(dt.string)),
    # Ceil
    ('ceil()', lambda x: x.ceil()),
    # Bucket
    ('bucket()', lambda x: x.bucket([0, 50, 100])),
    # Coalesce
    ('coalesce()', lambda x: x.coalesce()),
    # And
    # BitAnd
    ('bit_and()', lambda x: x.bit_and(True)),
    # BitOr
    ('bit_or()', lambda x: x.bit_or(True)),
    # BitXor
    ('bit_xor()', lambda x: x.bit_xor(True)),
    # CategoryLabel
    # CountDistinct
    ('count_distinct()', lambda x: x.count()),
    # Count
    ('count()', lambda x: x.count()),
    # CumulativeMax
    ('cummax()', lambda x: x.cummax()),
    # CumulativeMean
    ('cummean()', lambda x: x.cummean()),
    # CumulativeMin
    ('cummin()', lambda x: x.cummin()),
    # CumulativeSum
    ('cumsum()', lambda x: x.cumsum()),
    # DateAdd
    # DateDiff
    # DateFromYMD
    # DateSub
    # DateTruncate
    # Date
    # DayOfWeekIndex
    # DayOfWeekName
    # DenseRank
    ('dense_rank()', lambda x: x.dense_rank()),
    # EndsWith
    ('endswith()', lambda x: x.endswith('a')),
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
    ('first()', lambda x: x.first()),
    # FloorDivide
    # Floor
    ('floor()', lambda x: x.floor()),
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
    ('greatest()', lambda x: x.greatest(x.min())),
    # GroupConcat
    ('group_concat()', lambda x: x.group_concat()),
    # IdenticalTo
    # IfNull
    # IntervalFromInteger
    # IsNull
    # LPad
    # LStrip
    # Lag
    # LastValue
    # Lead
    ('lead()', lambda x: x.lead()),
    # Least
    ('least()', lambda x: x.least(x.max())),
    ('cumsum()', lambda x: x.cumsum()),
    # LessEqual
    # Less
    # Ln
    ('ln()', lambda x: x.ln()),
    # Log10
    ('log10()', lambda x: x.log10()),
    # Log2
    ('log2()', lambda x: x.log2()),
    # Log
    ('log()', lambda x: x.log(11)),
    # Lowercase
    # Max
    ('max()', lambda x: x.max()),
    # Mean
    ('mean()', lambda x: x.mean()),
    # MinRank
    # Min
    ('min()', lambda x: x.min()),
    # Modulus
    ('mod()', lambda x: x.mod(8)),
    # Multiply
    ('mul()', lambda x: x.mul(3)),
    # NTile
    ('ntile()', lambda x: x.ntile(2)),
    # Negate
    ('negate()', lambda x: x.negate()),
    # NotAll
    # NotAny
    # NotContains
    # NotEquals
    # NotNull
    ('notnull()', lambda x: x.notnull()),
    # Not
    # NullIfZero
    ('nullifzero()', lambda x: x.nullifzero()),
    # NullIf
    ('nullif()', lambda x: x.nullif(x.max())),
    # NullLiteral
    # Or
    # PercentRank
    ('percent_rank()', lambda x: x.percent_rank()),
    # Power
    # RPad
    # RStrip
    # RandomScalar
    # RegexSearch
    # Repeat
    # Reverse
    # Round
    ('round()', lambda x: x.round(4)),
    # RowNumber
    # SearchedCase
    # Sign
    ('sign()', lambda x: x.sign()),
    # SimpleCase
    # Sqrt
    ('sqrt()', lambda x: x.sqrt()),
    # StandardDev
    ('std()', lambda x: x.std()),
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
    ('sum()', lambda x: x.sum()),
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
    'bool_c',
]


@pytest.mark.parametrize(
    ('operation', 'column', 'ibis_operation'),
    [
        param(operation, column, ibis_operation, id=operation)
        for operation, ibis_operation in OPERATIONS
        for column in COLUMNS
    ],
)
def test_operations(con, operation, column, ibis_operation):
    datatypes = con.table('datatypes')

    try:
        ibis_operation(getattr(datatypes, column)).name('_').execute()
    except AttributeError:
        return
    print(operation, 'is supported for', column)
