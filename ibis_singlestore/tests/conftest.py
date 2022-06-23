#!/usr/bin/env python
"""SingleStore connection tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import ibis
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.base import RoundHalfToEven
from singlestore.connection import Connection

import pytest

ibis.options.interactive = True
ibis.options.sql.default_limit = None
ibis.options.verbose = True

USER = os.environ.get('IBIS_TEST_SINGLESTORE_USER', 'ibis')
PASSWORD = os.environ.get('IBIS_TEST_SINGLESTORE_PASSWORD', 'ibis')
HOST = os.environ.get('IBIS_TEST_SINGLESTORE_HOST', 'localhost')
PORT = os.environ.get('IBIS_TEST_SINGLESTORE_PORT', 3306)
DATABASE = os.environ.get('IBIS_TEST_SINGLESTORE_DATABASE', 'ibis_testing')

class TestConf(BackendTest, RoundHalfToEven):
    """
    SingleStore connection tests.

    Parameters
    ----------
    data_directory : Path
        Path to input data

    """

    # singlestore has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    bool_is_int = True

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)
        # mariadb supports window operations after version 10.2
        # but the sqlalchemy version string looks like:
        # 5.5.5.10.2.12.MariaDB.10.2.12+maria~jessie
        # or 10.4.12.MariaDB.1:10.4.12+maria~bionic
        # example of possible results:
        # https://github.com/sqlalchemy/sqlalchemy/blob/rel_1_3/
        # test/dialect/mysql/test_dialect.py#L244-L268
        self.__class__.supports_window_operations = True

    @staticmethod
    def connect(data_directory: Path) -> Connection:
        """
        Connect to SingleStore database.

        Parameters
        ----------
        data_directory : Path
            Path to input data

        Returns
        -------
        Connection

        """
        return ibis.singlestore.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
        )

def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{ibis.util.guid()}'

@pytest.fixture(scope='session')
def con():
    return ibis.singlestore.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
    )

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
    return con.table("intervals")


@pytest.fixture
def temp_table(con) -> Generator[str, None, None]:
    """
    Return a temporary table name.
    Parameters
    ----------
    con : ibis.postgres.PostgreSQLClient
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