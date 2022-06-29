from __future__ import annotations

import os
from pathlib import Path

import abc
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING, TextIO

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from filelock import FileLock
from toolz import dissoc

import ibis.expr.types as ir


import pytest
import sqlalchemy as sa
from packaging.version import parse as parse_version

import ibis

USER = os.environ.get('IBIS_TEST_SINGLESTORE_USER', 'ibis')
PASSWORD = os.environ.get('IBIS_TEST_SINGLESTORE_PASSWORD', 'ibis')
HOST = os.environ.get('IBIS_TEST_SINGLESTORE_HOST', 'localhost')
PORT = os.environ.get('IBIS_TEST_SINGLESTORE_PORT', 3306)
DATABASE = os.environ.get('IBIS_TEST_SINGLESTORE_DATABASE', 'ibis_testing')

@pytest.fixture(scope='session')
def data_directory() -> Path:
    """Return the test data directory.
    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parents[2]

    return Path(
        os.environ.get(
            "IBIS_TEST_DATA_DIRECTORY",
            root / "ci" / "ibis-testing-data",
        )
    )

TEST_TABLES = {
    "functional_alltypes": ibis.schema(
        {
            "index": "int64",
            "Unnamed: 0": "int64",
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        }
    ),
    "diamonds": ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
    ),
    "batting": ibis.schema(
        {
            "playerID": "string",
            "yearID": "int64",
            "stint": "int64",
            "teamID": "string",
            "lgID": "string",
            "G": "int64",
            "AB": "int64",
            "R": "int64",
            "H": "int64",
            "X2B": "int64",
            "X3B": "int64",
            "HR": "int64",
            "RBI": "int64",
            "SB": "int64",
            "CS": "int64",
            "BB": "int64",
            "SO": "int64",
            "IBB": "int64",
            "HBP": "int64",
            "SH": "int64",
            "SF": "int64",
            "GIDP": "int64",
        }
    ),
    "awards_players": ibis.schema(
        {
            "playerID": "string",
            "awardID": "string",
            "yearID": "int64",
            "lgID": "string",
            "tie": "string",
            "notes": "string",
        }
    ),
}

def recreate_database(driver, params, **kwargs):
    url = sa.engine.url.URL(driver, **dissoc(params, 'database'))
    engine = sa.create_engine(url, **kwargs)

    with engine.connect() as conn:
        conn.execute('DROP DATABASE IF EXISTS {}'.format(params['database']))
        conn.execute('CREATE DATABASE {}'.format(params['database']))

def init_database(
    url: sa.engine.url.URL,
    database: str,
    schema: TextIO | None = None,
    recreate: bool = True,
    **kwargs: Any,
) -> sa.engine.Engine:
    """Initialise {database} at {url} with {schema}.
    If {recreate}, drop the {database} at {url}, if it exists.
    Parameters
    ----------
    url : url.sa.engine.url.URL
        Connection url to the database
    database : str
        Name of the database to be dropped
    schema : TextIO
        File object containing schema to use
    recreate : bool
        If true, drop the database if it exists
    Returns
    -------
    sa.engine.Engine for the database created
    """
    if recreate:
        recreate_database(url, database, **kwargs)

    try:
        url.database = database
    except AttributeError:
        url = url.set(database=database)

    engine = sa.create_engine(url, **kwargs)

    if schema:
        with engine.connect() as conn:
            for stmt in filter(None, map(str.strip, schema.read().split(';'))):
                conn.execute(stmt)

    return engine

class RoundingConvention:
    @staticmethod
    @abc.abstractmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        """Round a series to `decimals` number of decimal values."""


# TODO: Merge into BackendTest, #2564
class RoundAwayFromZero(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        if not decimals:
            return (
                -(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)
            ).astype(np.int64)
        return series.round(decimals=decimals)

class RoundHalfToEven(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        result = series.round(decimals=decimals)
        return result if decimals else result.astype(np.int64)

class UnorderedComparator:
    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
        return super().assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        columns = list(set(left.columns) & set(right.columns))
        left = left.sort_values(by=columns)
        right = right.sort_values(by=columns)
        return super().assert_frame_equal(left, right, *args, **kwargs)

class BackendTest(abc.ABC):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    additional_skipped_operations = frozenset()
    supports_divide_by_zero = False
    returned_timestamp_unit = 'us'
    supported_to_timestamp_units = {'s', 'ms', 'us'}
    supports_floating_modulus = True
    bool_is_int = False
    supports_structs = True

    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)
        self.data_directory = data_directory

    def __str__(self):
        return f'<BackendTest {self.name()}>'

    @classmethod
    def name(cls) -> str:
        backend_tests_path = inspect.getmodule(cls).__file__
        return Path(backend_tests_path).resolve().parent.parent.parent.name[5:]

    @staticmethod
    @abc.abstractmethod
    def connect(data_directory: Path):
        """Return a connection with data loaded from `data_directory`."""

    @staticmethod
    def _load_data(
        data_directory: Path, script_directory: Path, **kwargs: Any
    ) -> None:
        ...

    @classmethod
    def load_data(
        cls,
        data_dir: Path,
        script_dir: Path,
        tmpdir: Path,
        worker_id: str,
        **kwargs: Any,
    ) -> None:
        """Load testdata from `data_directory` into
        the backend using scripts in `script_directory`."""
        # handling for multi-processes pytest

        # get the temp directory shared by all workers
        root_tmp_dir = tmpdir.getbasetemp()
        if worker_id != "master":
            root_tmp_dir = root_tmp_dir.parent

        fn = root_tmp_dir / f"lockfile_{cls.name()}"
        with FileLock(f"{fn}.lock"):
            if not fn.exists():
                cls._load_data(data_dir, script_dir, **kwargs)
                fn.touch()
        return cls(data_dir)

    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(
        series: pd.Series, name: str = 'tmp'
    ) -> pd.Series:
        return series.rename(name)

    @staticmethod
    def greatest(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        return f(*args)

    @staticmethod
    def least(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        return f(*args)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = self.connection.table('functional_alltypes')
        if self.bool_is_int:
            return t.mutate(bool_col=t.bool_col == 1)
        return t

    @property
    def batting(self) -> ir.Table:
        return self.connection.table('batting')

    @property
    def awards_players(self) -> ir.Table:
        return self.connection.table('awards_players')

    @property
    def geo(self) -> Optional[ir.Table]:
        if 'geo' in self.connection.list_tables():
            return self.connection.table('geo')
        return None

    @property
    def struct(self) -> Optional[ir.Table]:
        if self.supports_structs:
            return self.connection.table("struct")
        else:
            pytest.xfail(
                f"{self.name()} backend does not support struct types"
            )

    @property
    def api(self):
        return self.connection

    def make_context(self, params: Optional[Mapping[ir.Value, Any]] = None):
        return self.api.compiler.make_context(params=params)


class TestConf(BackendTest, RoundHalfToEven):
    # singlestore has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    bool_is_int = True
    supports_structs = False

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)
        # mariadb supports window operations after version 10.2
        # but the sqlalchemy version string looks like:
        # 5.5.5.10.2.12.MariaDB.10.2.12+maria~jessie
        # or 10.4.12.MariaDB.1:10.4.12+maria~bionic
        # example of possible results:
        # https://github.com/sqlalchemy/sqlalchemy/blob/rel_1_3/
        # test/dialect/singlestore/test_dialect.py#L244-L268
        con = self.connection
        if 'MariaDB' in str(con.version):
            # we might move this parsing step to the singlestore client
            version_detail = con.con.dialect._parse_server_version(
                str(con.version)
            )
            version = (
                version_detail[:3]
                if version_detail[3] == 'MariaDB'
                else version_detail[3:6]
            )
            self.__class__.supports_window_operations = version >= (10, 2)
        elif parse_version(con.version) >= parse_version('8.0'):
            # singlestore supports window operations after version 8
            self.__class__.supports_window_operations = True

    @staticmethod
    def _load_data(
        data_dir: Path,
        script_dir: Path,
        user: str = USER,
        password: str = PASSWORD,
        host: str = HOST,
        port: int = PORT,
        database: str = DATABASE,
        **_: Any,
    ) -> None:
        """Load test data into a SingleStore backend instance.
        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        with open(script_dir / 'schema' / 'singlestore.sql') as schema:
            engine = init_database(
                url=sa.engine.make_url(
                    f"singlestore+pymysql://{user}:{password}@{host}:{port:d}?local_infile=1",  # noqa: E501
                ),
                database=database,
                schema=schema,
                isolation_level="AUTOCOMMIT",
            )
            with engine.begin() as con:
                for table in TEST_TABLES:
                    csv_path = data_dir / f"{table}.csv"
                    lines = [
                        f"LOAD DATA LOCAL INFILE {str(csv_path)!r}",
                        f"INTO TABLE {table}",
                        "COLUMNS TERMINATED BY ','",
                        """OPTIONALLY ENCLOSED BY '"'""",
                        "LINES TERMINATED BY '\\n'",
                        "IGNORE 1 LINES",
                    ]
                    con.execute("\n".join(lines))

    @staticmethod
    def connect(_: Path):
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

@pytest.fixture(scope='session')
def backend():
    return TestConf(data_directory)