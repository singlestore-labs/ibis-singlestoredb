from __future__ import annotations

import os
from datetime import date
from operator import methodcaller
from typing import Any
from typing import Generator

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
import sqlalchemy_singlestoredb
from ibis.util import gen_name
from pytest import param


SINGLESTOREDB_DB = os.environ.get('IBIS_TEST_SINGLESTOREDB_DATABASE', 'ibis_testing')

SINGLESTOREDB_TYPES = [
    ('bool', dt.int8),
    ('boolean', dt.int8),
    ('bit', dt.int64),
    ('bit(1)', dt.int64),
    ('bit(9)', dt.int64),
    ('bit(17)', dt.int64),
    ('bit(33)', dt.int64),
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
    # TODO: The SET flag isn't sent to the client
    #   ("set('a', 'b', 'c', 'd')", dt.Array(dt.string)),
    ('bit(7)', dt.int64),
    # ('uuid', dt.string),
]


@pytest.mark.parametrize(
    ('singlestoredb_type', 'expected_type'),
    [
        param(singlestoredb_type, ibis_type, id=singlestoredb_type)
        for singlestoredb_type, ibis_type in SINGLESTOREDB_TYPES
    ],
)
def test_get_schema_from_query(
    con: Any,
    singlestoredb_type: Any,
    expected_type: Any,
) -> None:
    if 'unsigned' in singlestoredb_type and (
        'http://' in str(con.con.url) or 'https://' in str(con.con.url)
    ):
        pytest.skip('HTTP API does not surface unsigned int information')
        return

    raw_name = ibis.util.guid()
    name = con._quote(raw_name)
    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    with con.begin() as c:
        c.exec_driver_sql(
            f'CREATE ROWSTORE TEMPORARY TABLE {name} (x {singlestoredb_type})',
        )
    expected_schema = ibis.schema(dict(x=expected_type))
    t = con.table(raw_name)
    result_schema = con._get_schema_using_query(f'SELECT * FROM {name}')
    assert t.schema() == expected_schema
    assert result_schema == expected_schema


@pytest.mark.parametrize('coltype', ['TINYBLOB', 'MEDIUMBLOB', 'BLOB', 'LONGBLOB'])
def test_blob_type(con: Any, coltype: str) -> None:
    tmp = f'tmp_{ibis.util.guid()}'
    with con.begin() as c:
        c.exec_driver_sql(f'CREATE ROWSTORE TEMPORARY TABLE {tmp} (a {coltype})')
    t = con.table(tmp)
    assert t.schema() == ibis.schema({'a': dt.binary})


@pytest.fixture(scope='session')
def tmp_t(con_nodb: Any) -> Generator[Any, Any, Any]:
    pid = os.getpid()
    with con_nodb.begin() as c:
        c.exec_driver_sql(
            f'CREATE TABLE IF NOT EXISTS {SINGLESTOREDB_DB}.t_{pid} (x TEXT)',
        )
    yield
    with con_nodb.begin() as c:
        c.exec_driver_sql(f'DROP TABLE IF EXISTS {SINGLESTOREDB_DB}.t_{pid}')


@pytest.mark.usefixtures('setup_privs', 'tmp_t')
def test_get_schema_from_query_other_schema(con_nodb: Any) -> None:
    pid = os.getpid()
    t = con_nodb.table(f't_{pid}', schema=f'{SINGLESTOREDB_DB}')
    assert t.schema() == ibis.schema({'x': dt.string})


def test_zero_timestamp_data(con: Any) -> None:
    sql = """
    CREATE TEMPORARY TABLE ztmp_date_issue
    (
        name      CHAR(10) NULL,
        tradedate DATETIME NOT NULL,
        date      DATETIME NULL
    );
    """
    with con.begin() as c:
        c.exec_driver_sql(sql)
        c.exec_driver_sql(
            """
            INSERT INTO ztmp_date_issue VALUES
                ('C', '2018-10-22', 0),
                ('B', '2017-06-07', 0),
                ('A', '2022-12-21', 0)
            """,
        )
    t = con.table('ztmp_date_issue')
    result = t.execute().sort_values('name').reset_index(drop=True)
    expected = pd.DataFrame(
        {
            'name': ['C', 'B', 'A'],
            'tradedate': pd.to_datetime(
                [date(2018, 10, 22), date(2017, 6, 7), date(2022, 12, 21)],
            ),
            'date': [pd.NaT, pd.NaT, pd.NaT],
        },
    ).sort_values('name').reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.fixture(scope='module')
def enum_t(con: Any) -> Generator[Any, Any, Any]:
    name = gen_name('singlestoredb_enum_test')
    t = sa.Table(
        name, sa.MetaData(), sa.Column(
            'sml',
            sqlalchemy_singlestoredb.ENUM('small', 'medium', 'large'),
        ),
    )
    with con.begin() as bind:
        t.create(bind=bind)
        bind.execute(t.insert().values(sml='small'))

    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (methodcaller('startswith', 's'), pd.Series([True], name='sml')),
        (methodcaller('endswith', 'm'), pd.Series([False], name='sml')),
        (methodcaller('re_search', 'mall'), pd.Series([True], name='sml')),
        (methodcaller('lstrip'), pd.Series(['small'], name='sml')),
        (methodcaller('rstrip'), pd.Series(['small'], name='sml')),
        (methodcaller('strip'), pd.Series(['small'], name='sml')),
    ],
    ids=['startswith', 'endswith', 're_search', 'lstrip', 'rstrip', 'strip'],
)
def test_enum_as_string(enum_t: Any, expr_fn: Any, expected: Any) -> None:
    expr = expr_fn(enum_t.sml).name('sml')
    res = expr.execute()
    tm.assert_series_equal(res, expected)
