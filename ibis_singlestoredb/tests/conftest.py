#!/usr/bin/env python
# type: ignore
"""SingleStoreDB connection tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from typing import Iterable

import ibis
import pytest
from ibis.backends.tests.base import RoundHalfToEven
from ibis.backends.tests.base import ServiceBackendTest

ibis.options.interactive = False
ibis.options.sql.default_limit = None
ibis.options.verbose = False

SINGLESTOREDB_ROOT_USER = os.environ.get('IBIS_TEST_SINGLESTOREDB_ROOT_USER', 'root')
SINGLESTOREDB_ROOT_PASSWORD = os.environ.get('IBIS_TEST_SINGLESTOREDB_ROOT_PASSWORD', '')
SINGLESTOREDB_USER = os.environ.get('IBIS_TEST_SINGLESTOREDB_USER', 'ibis')
SINGLESTOREDB_PASSWORD = os.environ.get('IBIS_TEST_SINGLESTOREDB_PASSWORD', 'ibis')
SINGLESTOREDB_HOST = os.environ.get('IBIS_TEST_SINGLESTOREDB_HOST', 'localhost')
SINGLESTOREDB_PORT = int(os.environ.get('IBIS_TEST_SINGLESTOREDB_PORT', 9306))
SINGLESTOREDB_DB = os.environ.get('IBIS_TEST_SINGLESTOREDB_DATABASE', 'ibis_testing')

os.environ['SINGLESTOREDB_URL'] = \
    f'singlestoredb://{SINGLESTOREDB_USER}:{SINGLESTOREDB_PASSWORD}' \
    f'@{SINGLESTOREDB_HOST}:{SINGLESTOREDB_PORT}/{SINGLESTOREDB_DB}'


class TestConf(ServiceBackendTest, RoundHalfToEven):
    """
    SingleStoreDB connection tests.

    Parameters
    ----------
    data_directory : Path
        Path to input data

    """

    # singlestoredb has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = 's'
    supports_window_operations = True
    supports_arrays = False
    supports_structs = False
    supports_arrays_outside_of_select = supports_arrays
    bool_is_int = True

    service_name = 'singlestoredb'
    deps = 'singlestoredb', 'sqlalchemy-singlestoredb', 'sqlalchemy'

    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath('csv').glob('*.csv')

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        database = kw.pop('database', SINGLESTOREDB_DB)
        return ibis.singlestoredb.connect(
            host=SINGLESTOREDB_HOST,
            user=SINGLESTOREDB_USER,
            password=SINGLESTOREDB_PASSWORD,
            database=database,
            port=SINGLESTOREDB_PORT,
            **kw,
        )


def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{ibis.util.guid()}'


@pytest.fixture(scope='session')
def setup_privs():
    #    engine = sa.create_engine(
    #        f'singlestoredb://{SINGLESTOREDB_ROOT_USER}:{SINGLESTOREDB_ROOT_PASSWORD}'
    #        f'@{SINGLESTOREDB_HOST}:{SINGLESTOREDB_PORT}',
    #    )
    #    with engine.begin() as con:
    #        # allow the ibis user to use any database
    #        con.exec_driver_sql(f'CREATE DATABASE IF NOT EXISTS `{SINGLESTOREDB_DB}`')
    #        con.exec_driver_sql(
    #            f'GRANT CREATE,SELECT,DROP ON `{SINGLESTOREDB_DB}`.* '
    #            f'TO `{SINGLESTOREDB_USER}`@`%%`',
    #        )
    yield
#    with engine.begin() as con:
#        con.exec_driver_sql(f'DROP DATABASE IF EXISTS `{SINGLESTOREDB_DB}`')


@pytest.fixture(scope='session')
def con():
    # sql_file = os.path.join(os.path.dirname(__file__), 'test.sql')
    # dbname, dbexisted = utils.load_sql(sql_file)

    yield TestConf.connect(tmpdir=None, worker_id=None)

    # if not dbexisted:
    #     utils.drop_database(dbname)


@pytest.fixture(scope='session')
def con_nodb():
    return TestConf.connect(tmpdir=None, worker_id=None, database=None)


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def geotable(con):
    return con.table('geo')


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='module')
def gdf(geotable):
    return geotable.execute()


@pytest.fixture(scope='module')
def at(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope='module')
def intervals(con):
    return con.table('intervals')


@pytest.fixture
def temp_table(con) -> Generator[str, None, None]:
    """
    Return a temporary table name.

    Parameters
    ----------
    con : ibis.singlestoredb.SingleStoreDBClient

    Yields
    ------
    name : string
        Random table name for a temporary usage.

    """
    name = _random_identifier('table')
    try:
        yield name
    finally:
        con.drop_table(name, force=True)
