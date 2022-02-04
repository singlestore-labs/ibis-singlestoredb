# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import base64
import json
import string
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Sequence

import ibis.expr.schema as sch
from ibis.backends.base.sql.ddl import AlterTable
from ibis.backends.base.sql.ddl import BaseDDL
from ibis.backends.base.sql.ddl import CreateTable
from ibis.backends.base.sql.ddl import CreateTableWithSchema
from ibis.backends.base.sql.ddl import DropFunction as DDLDropFunction
from ibis.backends.base.sql.ddl import format_partition
from ibis.backends.base.sql.ddl import format_schema
from ibis.backends.base.sql.ddl import format_tblproperties
from ibis.backends.base.sql.registry import type_to_sql_string

from .udf import SingleStoreUDF


class CreateTableParquet(CreateTable):
    def __init__(
        self,
        table_name: str,
        path: str,
        example_file: Optional[str] = None,
        example_table: Optional[str] = None,
        schema: Optional[sch.Schema] = None,
        external: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            table_name,
            external=external,
            format='parquet',
            path=path,
            **kwargs,
        )
        self.example_file = example_file
        self.example_table = example_table
        self.schema = schema

    @property
    def _pieces(self) -> Iterator[str]:
        if self.example_file is not None:
            yield f"LIKE PARQUET '{self.example_file}'"
        elif self.example_table is not None:
            yield f'LIKE {self.example_table}'
        elif self.schema is not None:
            yield format_schema(self.schema)
        else:
            raise NotImplementedError

        yield self._storage()
        yield self._location()


class DelimitedFormat:
    def __init__(
        self,
        path: str,
        delimiter: Optional[str] = None,
        escapechar: Optional[str] = None,
        na_rep: Optional[str] = None,
        lineterminator: Optional[str] = None,
    ):
        self.path = path
        self.delimiter = delimiter
        self.escapechar = escapechar
        self.lineterminator = lineterminator
        self.na_rep = na_rep

    def to_ddl(self) -> Iterator[str]:
        yield 'ROW FORMAT DELIMITED'

        if self.delimiter is not None:
            yield f"FIELDS TERMINATED BY '{self.delimiter}'"

        if self.escapechar is not None:
            yield f"ESCAPED BY '{self.escapechar}'"

        if self.lineterminator is not None:
            yield f"LINES TERMINATED BY '{self.lineterminator}'"

        yield f"LOCATION '{self.path}'"

        if self.na_rep is not None:
            props = {'serialization.null.format': self.na_rep}
            yield format_tblproperties(props)


class AvroFormat:

    def __init__(self, path: str, avro_schema: str):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self) -> Iterator[str]:
        yield 'STORED AS AVRO'
        yield f"LOCATION '{self.path}'"

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = '\n'.join(x.rstrip() for x in schema.splitlines())

        props = {'avro.schema.literal': schema}
        yield format_tblproperties(props)


class ParquetFormat:
    def __init__(self, path: str) -> None:
        self.path = path

    def to_ddl(self) -> Iterator[str]:
        yield 'STORED AS PARQUET'
        yield f"LOCATION '{self.path}'"


class CreateTableDelimited(CreateTableWithSchema):

    def __init__(
        self,
        table_name: str,
        path: str,
        schema: sch.Schema,
        delimiter: Optional[str] = None,
        escapechar: Optional[str] = None,
        lineterminator: Optional[str] = None,
        na_rep: Optional[str] = None,
        external: bool = True,
        **kwargs: Any,
    ):
        table_format = DelimitedFormat(
            path,
            delimiter=delimiter,
            escapechar=escapechar,
            lineterminator=lineterminator,
            na_rep=na_rep,
        )
        super().__init__(
            table_name, schema, table_format, external=external, **kwargs,
        )


class CreateTableAvro(CreateTable):

    def __init__(
        self,
        table_name: str,
        path: str,
        avro_schema: str,
        external: bool = True,
        **kwargs: Any,
    ):
        super().__init__(table_name, external=external, **kwargs)
        self.table_format = AvroFormat(path, avro_schema)

    @property
    def _pieces(self) -> Iterator[str]:
        yield '\n'.join(self.table_format.to_ddl())


class LoadData(BaseDDL):
    """
    Generate DDL for LOAD DATA command. Cannot be cancelled
    """

    def __init__(
        self,
        table_name: str,
        path: str,
        database: Optional[str] = None,
        partition: Optional[str] = None,
        partition_schema: Optional[str] = None,
        overwrite: bool = False,
    ):
        self.table_name = table_name
        self.database = database
        self.path = path

        self.partition = partition
        self.partition_schema = partition_schema

        self.overwrite = overwrite

    def compile(self) -> str:
        overwrite = 'OVERWRITE ' if self.overwrite else ''

        if self.partition is not None:
            partition = '\n' + format_partition(
                self.partition, self.partition_schema,
            )
        else:
            partition = ''

        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return "LOAD DATA INPATH '{}' {}INTO TABLE {}{}".format(
            self.path, overwrite, scoped_name, partition,
        )


class PartitionProperties(AlterTable):

    def __init__(
        self,
        table: str,
        partition: str,
        partition_schema: str,
        location: Optional[str] = None,
        format: Optional[str] = None,
        tbl_properties: Optional[dict[str, Any]] = None,
        serde_properties: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            table,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )
        self.partition = partition
        self.partition_schema = partition_schema

    def _compile(self, cmd: str, property_prefix: str = '') -> str:
        part = format_partition(self.partition, self.partition_schema)
        if cmd:
            part = f'{cmd} {part}'

        props = self._format_properties(property_prefix)
        action = f'{self.table} {part}{props}'
        return self._wrap_command(action)


class AddPartition(PartitionProperties):

    def __init__(
        self,
        table: str,
        partition: str,
        partition_schema: str,
        location: Optional[str] = None,
    ):
        super().__init__(table, partition, partition_schema, location=location)

    def compile(self) -> str:
        return self._compile('ADD')


class AlterPartition(PartitionProperties):

    def compile(self) -> str:
        return self._compile('', 'SET ')


class DropPartition(PartitionProperties):

    def __init__(self, table: str, partition: str, partition_schema: str):
        super().__init__(table, partition, partition_schema)

    def compile(self) -> str:
        return self._compile('DROP')


class CacheTable(BaseDDL):

    def __init__(
        self,
        table_name: str,
        database: Optional[str] = None,
        pool: str = 'default',
    ):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self) -> str:
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return "ALTER TABLE {} SET CACHED IN '{}'".format(
            scoped_name, self.pool,
        )


class CreateFunction(BaseDDL):

    _object_type = 'FUNCTION'

    def __init__(
        self,
        func: SingleStoreUDF,
        name: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.name
        self.database = database

    def _singlestore_signature(self) -> str:
        scoped_name = self._get_scoped_name(self.func.symbol, self.database)
        input_sig = _singlestore_input_signature(self.func.inputs)
        output_sig = type_to_sql_string(self.func.output)

        return f'{scoped_name}({input_sig}) RETURNS {output_sig} NOT NULL'


class CreateUDF(CreateFunction):

    def compile(self) -> str:
        create_decl = 'CREATE OR REPLACE FUNCTION'
        singlestore_sig = self._singlestore_signature()

        if self.func.language == self.func.LANGUAGE_WASM:
            param_line = 'AS WASM'
        elif self.func.language == self.func.LANGUAGE_PYTHON:
            param_line = 'AS PYTHON'
        else:
            raise ValueError(f"Unsupported function language: {self.func.library!r}'")

        if self.func.type == self.func.TYPE_FILE:
            param_line += f" INFILE '{self.func.library!r}'"
        elif self.func.type == self.func.TYPE_MODULE:
            library = bytes('{!r}'.format(self.func.library or ''), 'utf-8')
            param_line += " '{}'".format(base64.b64encode(library).decode('utf-8'))
        else:
            raise ValueError('Unsupported function format')

        return ' '.join([create_decl, singlestore_sig, param_line])


class CreateUDA(CreateFunction):

    def compile(self) -> str:
        create_decl = 'CREATE OR REPLACE AGGREGATE FUNCTION'
        singlestore_sig = self._singlestore_signature()
        tokens = [f"AS INFILE '{self.func.library!r}'"]

        fn_names = (
            'init_fn',
            'update_fn',
            'merge_fn',
            'serialize_fn',
            'finalize_fn',
        )

        for fn in fn_names:
            value = getattr(self.func, fn)
            if value is not None:
                tokens.append(f"{fn}='{value}'")

        return ' '.join([create_decl, singlestore_sig]) + ' ' + '\n'.join(tokens)


class DropFunction(DDLDropFunction):

    def _singlestore_signature(self) -> str:
        full_name = self._get_scoped_name(self.name, self.database)
        input_sig = _singlestore_input_signature(self.inputs)
        return f'{full_name}({input_sig})'


class ListFunction(BaseDDL):

    def __init__(
        self,
        database: str,
        like: Optional[str] = None,
        aggregate: bool = False,
    ):
        self.database = database
        self.like = like
        self.aggregate = aggregate

    def compile(self) -> str:
        statement = 'SHOW '
        if self.aggregate:
            statement += 'AGGREGATE '
        statement += f'FUNCTIONS IN {self.database}'
        if self.like:
            statement += f" LIKE '{self.like}'"
        return statement


def _singlestore_input_signature(inputs: Sequence[str]) -> str:
    # TODO: varargs '{}...'.format(val)
    return ', '.join([
        '{} {} NOT NULL'.format(
            string.ascii_letters[i],
            type_to_sql_string(x),
        ) for i, x in enumerate(inputs)
    ])
