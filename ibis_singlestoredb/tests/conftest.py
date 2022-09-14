#!/usr/bin/env python
# type: ignore
"""SingleStoreDB connection tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import ibis
import pytest
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.base import RoundHalfToEven
from singlestoredb.connection import Connection
from singlestoredb.tests import utils

ibis.options.interactive = False
ibis.options.sql.default_limit = None
ibis.options.verbose = False


class TestConf(BackendTest, RoundHalfToEven):
    """
    SingleStoreDB connection tests.

    Parameters
    ----------
    data_directory : Path
        Path to input data

    """

    # singlestoredb has the same rounding behavior as postgres
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
        Connect to SingleStoreDB database.

        Parameters
        ----------
        data_directory : Path
            Path to input data

        Returns
        -------
        Connection

        """
        return ibis.singlestoredb.connect()


def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{ibis.util.guid()}'


@pytest.fixture(scope='session')
def con():
    sql_file = os.path.join(os.path.dirname(__file__), 'test.sql')
    dbname, dbexisted = utils.load_sql(sql_file)

    yield ibis.singlestoredb.connect(database=dbname)

    if not dbexisted:
        utils.drop_database(dbname)


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
