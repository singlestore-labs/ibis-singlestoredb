"""The SingleStore backend."""

from __future__ import annotations

import atexit
import contextlib
import warnings
from typing import Literal

import sqlalchemy as sa
import sqlalchemy.dialects.mysql as singlestore

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from .compiler import SingleStoreCompiler
from .datatypes import _type_from_cursor_info

from singlestore.connection import build_params

class Backend(BaseAlchemyBackend):
    name = 'singlestore'
    compiler = SingleStoreCompiler

    def do_connect(self, *args: str, **kwargs: Any) -> None:
        """Connect to a SingleStore database."""
        if args:
            params = build_params(host=args[0], **kwargs)
        else:
            params = build_params(**kwargs)

        driver = params.pop('driver', None)
        if driver and not driver.startswith('singlestore+'):
            driver = 'singlestore+{}'.format(driver)

        alchemy_url = self._build_alchemy_url(
            url=params.pop('url', None),
            host=params.pop('host', None),
            port=params.pop('port', None),
            user=params.pop('user', None),
            password=params.pop('password', None),
            database=params.pop('database', None),
            driver=driver,
        )

        alchemy_url.set(query={k: str(v) for k, v in params.items()})

        self.database_name = alchemy_url.database

        super().do_connect(
            sa.create_engine(
                alchemy_url,
                echo=kwargs.get('echo', False), future=kwargs.get('future', False),
            ),
        )

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute(
                'SELECT @@session.time_zone'
            ).scalar()
            try:
                bind.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Couldn't set singlestore timezone: {str(e)}")

            try:
                yield bind
            finally:
                query = "SET @@session.time_zone = '{}'"
                bind.execute(query.format(previous_timezone))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Infer the schema of `query`."""
        result = self.con.execute(f"SELECT * FROM ({query}) _ LIMIT 0")
        cursor = result.cursor
        fields = [
            (descr[0], _type_from_cursor_info(descr))
            for descr in cursor.description
        ]
        return sch.Schema.from_tuples(fields)

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f"CREATE OR REPLACE VIEW {name} AS {definition}"

    def _register_temp_view_cleanup(self, name: str, raw_name: str) -> None:
        query = f"DROP VIEW IF EXISTS {name}"

        def drop(self, raw_name: str, query: str):
            self.con.execute(query)
            self._temp_views.discard(raw_name)

        atexit.register(drop, self, raw_name, query)


# TODO(kszucs): unsigned integers


@dt.dtype.register((singlestore.DOUBLE, singlestore.REAL))
def singlestore_double(satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(singlestore.FLOAT)
def singlestore_float(satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(singlestore.TINYINT)
def singlestore_tinyint(satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(singlestore.BLOB)
def singlestore_blob(satype, nullable=True):
    return dt.Binary(nullable=nullable)

@dt.dtype.register(singlestore.BIT)
def singlestore_bit(satype, nullable=True):
    """
    Register a bit data type.

    Parameters
    ----------
    satype : Any
        SQLAlchemy data type
    nullable : bool, optional
        Is the column nullable?

    Returns
    -------
    dt.DataType

    """
    return dt.Binary(nullable=nullable)