#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import pytest
from pytest import param
from sqlalchemy import false
from sqlalchemy import true

SINGLESTOREDB_TYPES = [
    ('bool', dt.int8),
    ('boolean', dt.int8),
    ('bit', dt.int64),
    #   ("bit(1)", dt.int64),
    #   ("bit(9)", dt.int64),
    #   ("bit(17)", dt.int64),
    #   ("bit(33)", dt.int64),
    ('tinyint', dt.int8),
    ('tinyint unsigned', dt.uint8),
    ('smallint', dt.int16),
    ('smallint unsigned', dt.uint16),
    ('mediumint', dt.int32),
    ('mediumint unsigned', dt.uint32),
    ('int', dt.int32),
    ('int unsigned', dt.uint32),
    ('integer', dt.int32),
    ('integer unsigned', dt.uint32),
    ('bigint', dt.int64),
    ('bigint unsigned', dt.uint64),
    ('int1', dt.int8),
    ('int1 unsigned', dt.uint8),
    ('int2', dt.int16),
    ('int2 unsigned', dt.uint16),
    ('int3', dt.int32),
    ('int3 unsigned', dt.uint32),
    ('int4 unsigned', dt.uint32),
    ('float', dt.float32),
    ('double', dt.float64),
    ('decimal(10, 0)', dt.Decimal(10, 0)),
    ('decimal(5, 2)', dt.Decimal(5, 2)),
    ('dec', dt.Decimal(10, 0)),
    ('dec(5, 2)', dt.Decimal(5, 2)),
    ('numeric', dt.Decimal(10, 0)),
    ('fixed', dt.Decimal(10, 0)),
    ('date', dt.date),
    ('time', dt.time),
    ('time(6)', dt.time),
    ('datetime', dt.timestamp),
    ('datetime(6)', dt.timestamp),
    ('timestamp', dt.Timestamp('UTC')),
    ('timestamp(6)', dt.Timestamp('UTC')),
    ('year', dt.uint8),
    ('char(32)', dt.string),
    ('binary(42)', dt.binary),
    ('varchar(42)', dt.string),
    ('char byte', dt.binary),
    ('text', dt.string),
    ('varbinary(42)', dt.binary),
    ('longtext', dt.string),
    ('mediumtext', dt.string),
    ('text', dt.string),
    ('tinytext', dt.string),
    ('text(4)', dt.string),
    ('longblob', dt.binary),
    ('mediumblob', dt.binary),
    ('blob', dt.binary),
    ('tinyblob', dt.binary),
    ('json', dt.json),
    ('geographypoint', dt.string),
    ('geography', dt.string),
    ("enum('small', 'medium', 'large')", dt.string),
    ("set('a', 'b', 'c', 'd')", dt.string),
    #   ("bit(7)", dt.binary)
]


@pytest.mark.parametrize(
    ('singlestoredb_type', 'expected_type'),
    [
        param(singlestoredb_type, ibis_type, id=singlestoredb_type)
        for singlestoredb_type, ibis_type in SINGLESTOREDB_TYPES
    ],
)
def test_get_schema_from_query(con, singlestoredb_type, expected_type):
    if 'unsigned' in singlestoredb_type and \
            ('http://' in str(con.con.url) or 'https://' in str(con.con.url)):
        pytest.skip('HTTP API does not surface unsigned int information')
        return

    raw_name = ibis.util.guid()
    name = con.con.dialect.identifier_preparer.quote_identifier(raw_name)
    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    con.raw_sql(
        f'CREATE ROWSTORE TEMPORARY TABLE {name} '
        f'(x {singlestoredb_type}, y {singlestoredb_type})',
    )
    expected_schema = ibis.schema(dict(x=expected_type, y=expected_type))
    result_schema = con._get_schema_using_query(f'SELECT * FROM {name}')
    assert are_equal(
        result_schema, expected_schema,
    ), f'{result_schema} : {expected_schema}'

    if singlestoredb_type != 'geography':
        raw_name = ibis.util.guid()
        name = con.con.dialect.identifier_preparer.quote_identifier(raw_name)
        con.raw_sql(
            f'CREATE TEMPORARY TABLE {name} '
            f'(x {singlestoredb_type}, y {singlestoredb_type})',
        )
        expected_schema = ibis.schema(dict(x=expected_type, y=expected_type))
        result_schema = con._get_schema_using_query(f'SELECT * FROM {name}')
        assert are_equal(
            result_schema, expected_schema,
        ), f'{result_schema} : {expected_schema}'


def are_equal(result_schema, expected_schema):
    name_check = result_schema.names == expected_schema.names
    type_check = true

    if result_schema.types == (
        dt.Decimal(precision=None, scale=None, nullable=True),
        dt.Decimal(precision=None, scale=None, nullable=True),
    ) or \
            result_schema.types == (dt.Int64(nullable=True), dt.Int64(nullable=True)):

        for i, stype in enumerate(expected_schema.types):
            for name, expected_value in zip(stype.argnames, stype.args):
                result_value = getattr(result_schema.types[i], name, None)
                if result_value is not None and expected_value != result_value:
                    type_check = false
    else:
        type_check = result_schema.types == expected_schema.types

    return name_check and type_check
