"""The SingleStore backend."""

from __future__ import annotations

import atexit
import contextlib
import warnings
from typing import Any, Literal

import sqlalchemy as sa
import sqlalchemy.dialects.mysql as singlestore
from sqlalchemy_singlestore.base import SingleStoreDialect

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.common.exceptions as com
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from .compiler import SingleStoreCompiler
from .datatypes import _type_from_cursor_info

import pandas as pd
from pandas.io.json import build_table_schema
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
    
    def table(
        self,
        name: str,
        database: str | None = None,
        schema: str | None = None,
    ) -> ir.TableExpr:
        """Create a table expression from a table in the database.

        Parameters
        ----------
        name
            Table name
        database
            The database the table resides in
        schema
            The schema inside `database` where the table resides.

            !!! warning "`schema` refers to database organization"

                The `schema` parameter does **not** refer to the column names
                and types of `table`.

        Returns
        -------
        TableExpr
            Table expression
        """
        if database is not None and database != self.current_database:
            return self.database(database=database).table(
                name=name,
                database=database,
                schema=schema,
            )
        sqla_table = self._get_sqla_table(
            name,
            database=database,
            schema=self._current_schema,
        )
        return self._sqla_table_to_expr(sqla_table)

    def create_table(
        self,
        name: str,
        expr: pd.DataFrame | ir.TableExpr | None = None,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False
    ) -> None:
        """Create a new table.
        Parameters
        ----------
        name
            Name of the new table.
        expr
            An Ibis table expression or pandas table that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `expr` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.
        """
        if expr is None and schema is None:
            raise ValueError('You must pass either an expression or a schema')

        if isinstance(expr, pd.DataFrame):
            expr_schema = ibis.pandas.connect({name : expr}).table(name).schema()
        elif isinstance(expr, ir.TableExpr):
            expr_schema = expr.schema()
            expr = expr.execute()

        if expr is not None and schema is not None:
            if not expr_schema.equals(schema):
                raise TypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )

        self._schemas[self._fully_qualified_name(name, database)] = expr_schema
        self._table_from_schema(
            name, expr_schema, database=database or self.current_database
        )

        expr.to_sql(
                name,
                self.con,
                index=False,
                if_exists='replace' if force else 'append',
                schema=self._current_schema,
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


@dt.dtype.register(SingleStoreDialect, singlestore.BIT)
def singlestore_bit(dialect, satype, nullable=True):
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