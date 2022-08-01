#!/usr/bin/env python
# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for defining SingleStore UDFs / UDAs."""
from __future__ import annotations

import abc
import os
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.udf.validate as v
import ibis.util as util
from ibis.backends.base.sql.registry import sql_type_names

from .compiler import SingleStoreExprTranslator
from .registry import fixed_arity

__all__ = [
    'add_operation',
    'scalar_function',
    'aggregate_function',
    'wrap_udf',
    'wrap_uda',
]


class Function(metaclass=abc.ABCMeta):
    """
    Base class for UDFs / UDAs.

    Parameters
    ----------
    inputs : Sequence[str]
        Input parameter type names
    output : str
        Output value type name
    name : str, optional
        Name of the function

    """

    def __init__(self, inputs: Sequence[str], output: str, name: Optional[str] = None):
        self.inputs = tuple(map(dt.dtype, inputs))
        self.output = dt.dtype(output)
        self.name = name or util.guid()
        self._klass = self._create_operation_class()

    @abc.abstractmethod
    def _create_operation_class(self) -> ops.ValueOp:
        """
        Create a new class for the operation.

        Returns
        -------
        ops.ValueOp

        """
        pass

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str

        """
        klass = type(self).__name__
        return '{}({}, {!r}, {!r})'.format(
            klass, self.name, self.inputs, self.output,
        )

    def __call__(self, *args: Any) -> ir.Expr:
        """
        Call the function.

        Parameters
        ----------
        *args : positional arguments, optional
            Arguments to the function

        Returns
        -------
        ir.Expr

        """
        return self._klass(*args).to_expr()

    def register(self, name: str, database: str) -> None:
        """
        Register given operation with the Ibis SQL translation toolchain.

        Parameters
        ----------
        name : str
            Name of the function
        database : str
            Name of the database to register the function with

        Returns
        -------
        None

        """
        add_operation(self._klass, name, database)


class ScalarFunction(Function):
    """Base class for UDFs."""

    def _create_operation_class(self) -> ops.ValueOp:
        """
        Create a new class for the UDF operation.

        Returns
        -------
        ops.ValueOp

        """
        fields = {
            f'_{i}': rlz.value(dtype) for i, dtype in enumerate(self.inputs)
        }
        fields['output_type'] = rlz.shape_like('args', self.output)
        return type(f'UDF_{self.name}', (ops.ValueOp,), fields)


class AggregateFunction(Function):
    """Base class for UDAs."""

    def _create_operation_class(self) -> ops.ValueOp:
        """
        Create a new class for the UDA operation.

        Returns
        -------
        ops.ValueOp

        """
        fields = {
            f'_{i}': rlz.value(dtype) for i, dtype in enumerate(self.inputs)
        }
        fields['output_type'] = lambda op: self.output.scalar_type()
        fields['_reduction'] = True
        return type(f'UDA_{self.name}', (ops.ValueOp,), fields)


class SingleStoreFunction:
    """
    Base class for SingleStore UDFs / UDAs.

    Parameters
    ----------
    name : str, optional
        Name of the function
    library : str or bytes, optional
        Path to or content of function library

    """

    TYPE_UNKNOWN = 0
    TYPE_FILE = 1
    TYPE_MODULE = 2

    LANGUAGE_UNKNOWN = 0
    LANGUAGE_WASM = 1
    LANGUAGE_PYTHON = 2

    def __init__(
        self,
        name: Optional[str] = None,
        library: Optional[Union[str, bytes]] = None,
    ):
        self.library = library
        self.name = name or util.guid()
        self.type = self.TYPE_UNKNOWN
        self.language = self.LANGUAGE_UNKNOWN

        if library is not None:
            self._check_library()

    def _check_library(self) -> None:
        """
        Introspect the specified library.

        This function determines what type of function is contained
        in the specified library.

        Returns
        -------
        None

        """
        if self.library is None:
            pass

        elif os.path.isfile(self.library):
            self.library = str(self.library)
            parts = self.library.split('.')
            if len(parts) < 2 or parts[-1] not in ['wasm', 'py']:
                raise ValueError("Invalid file type. Must be '.wasm' or '.py'.")
            self.language = parts[-1] == 'wasm' and \
                self.LANGUAGE_WASM or self.LANGUAGE_PYTHON
            self.type = self.TYPE_FILE

        elif isinstance(self.library, str) and self.library[:4] == '\x00asm':
            raise TypeError('WASM modules must be bytes-like.')

        elif self.library[:4] == b'\x00asm':
            self.type = self.TYPE_MODULE
            self.language = self.LANGUAGE_WASM

        elif isinstance(self.library, str) and 'def ' in self.library:
            self.type = self.TYPE_MODULE
            self.language = self.LANGUAGE_PYTHON
            self.library = self.library.encode('utf-8')

        elif isinstance(self.library, bytes) and b'def' in self.library:
            self.type = self.TYPE_MODULE
            self.language = self.LANGUAGE_PYTHON

        else:
            raise ValueError('Could not determine library type.')

    def hash(self) -> None:
        """Return a unique hash."""
        raise NotImplementedError


class SingleStoreUDF(ScalarFunction, SingleStoreFunction):
    """
    Base class for SingleStore UDFs.

    Parameters
    ----------
    inputs : Sequence[str]
        List of input parameter type names
    output : str
        Data type of function return value
    symbol : str
        Name of the function symbol in the library
    library : str
        Path to or content of function library
    name : str
        Name to apply to the function

    """

    def __init__(
        self,
        inputs: Sequence[str],
        output: str,
        symbol: str,
        library: Optional[str] = None,
        name: Optional[str] = None,
    ):
        v.validate_output_type(output)
        self.symbol = symbol
        SingleStoreFunction.__init__(self, name=name, library=library)
        ScalarFunction.__init__(self, inputs, output, name=self.symbol)

    def hash(self) -> None:
        """Return a unique hash."""
        # TODO: revisit this later
        # from hashlib import sha1
        # val = self.symbol
        # for in_type in self.inputs:
        #     val += in_type.name()

        # return sha1(val).hexdigest()
        pass


class SingleStoreUDA(AggregateFunction, SingleStoreFunction):
    """
    Base class for SingleStore UDAs.

    Parameters
    ----------
    inputs : Sequence[str]
        List of input parameter type names
    output : str
        Data type of function return value
    update_fn : str
        Name of the update function
    init_fn : str
        Name of the initialization function
    merge_fn : str
        Name of the merge function
    serialize_fn : str
        Name of the serialization function
    library : str
        Path to or content of the function library
    name : str
        Name to apply to the function

    """

    def __init__(
        self,
        inputs: Sequence[str],
        output: str,
        update_fn: Optional[str] = None,
        init_fn: Optional[str] = None,
        merge_fn: Optional[str] = None,
        finalize_fn: Optional[str] = None,
        serialize_fn: Optional[str] = None,
        library: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn
        self.serialize_fn = serialize_fn

        v.validate_output_type(output)

        SingleStoreFunction.__init__(self, name=name, library=library)
        AggregateFunction.__init__(self, inputs, output, name=self.name)

    def _check_library(self) -> None:
        """
        Determine the type of the given library.

        Returns
        -------
        None

        """
        if self.library is None:
            raise ValueError('No library was specified')
        suffix = self.library[-3:]
        if suffix == '.ll':
            raise com.IbisInputError('LLVM IR UDAs are not yet supported')
        elif suffix != '.so':
            raise ValueError('Invalid file type. Must be .so')


def wrap_uda(
    library: str,
    inputs: Sequence[str],
    output: str,
    update_fn: str,
    init_fn: Optional[str] = None,
    merge_fn: Optional[str] = None,
    finalize_fn: Optional[str] = None,
    serialize_fn: Optional[str] = None,
    close_fn: Optional[str] = None,
    name: Optional[str] = None,
) -> SingleStoreUDA:
    """
    Create a callable aggregation function object.

    Parameters
    ----------
    library : str
        Path to or content of function library
    inputs : Sequence[str]
        List of input data type names
    output : str
        Data type name of function return value
    update_fn : string
        Library symbol name for update function
    init_fn : string, optional
        Library symbol name for initialization function
    merge_fn : string, optional
        Library symbol name for merge function
    finalize_fn : string, optional
        Library symbol name for finalize function
    serialize_fn : string, optional
        Library symbol name for serialize UDA API function. Not required for all
        UDAs; see documentation for more.
    close_fn : string, optional
        Exit function
    name: string, optional
        Used internally to track function

    Returns
    -------
    SingleStoreUDA

    """
    func = SingleStoreUDA(
        inputs,
        output,
        update_fn,
        init_fn,
        merge_fn,
        finalize_fn,
        serialize_fn=serialize_fn,
        name=name,
        library=library,
    )
    return func


def wrap_udf(
    library: str,
    inputs: Sequence[str],
    output: str,
    symbol: str,
    name: Optional[str] = None,
) -> SingleStoreUDF:
    """
    Create a callable scalar function object.

    Parameters
    ----------
    library : str
        Path to or content of function library
    inputs : Sequence[str]
        Input parameter type names
    output : string
        Data type of the return value of the function
    symbol : string
        Symbol name in the library
    name : str, optional
        Used internally to track function

    Returns
    -------
    SingleStoreUDF

    """
    func = SingleStoreUDF(inputs, output, symbol, name=name, library=library)
    return func


def scalar_function(
    inputs: Sequence[str],
    output: str,
    name: Optional[str] = None,
) -> ScalarFunction:
    """
    Create an operator class that can be passed to `add_operation`.

    Parameters
    ----------
    inputs : Sequence[str]
        Input parameter type names
    output : str
        Data type of the return value of the function
    name : str, optional
      Used internally to track function

    Returns
    -------
    ScalarFunction

    """
    return ScalarFunction(inputs, output, name=name)


def aggregate_function(
    inputs: Sequence[str],
    output: str,
    name: Optional[str] = None,
) -> AggregateFunction:
    """
    Create an operator class that can be passed to `add_operation`.

    Parameters
    ----------
    inputs : Sequnce[str]
        Input parameter type names
    output : str
        Data type of the return value of the function
    name : str, optional
        Used internally to track function

    Returns
    -------
    AggregateFunction

    """
    return AggregateFunction(inputs, output, name=name)


def add_operation(op: ops.ValueOp, func_name: str, db: str) -> ops.ValueOp:
    """
    Register the given operation with the Ibis SQL translation toolchain.

    Parameters
    ----------
    op : ops.ValueOp
        Operator class
    name : str
        Name of the operation
    database : str
        Database to register the operation in

    Returns
    -------
    None

    """
    # TODO
    # if op.input_type is rlz.listof:
    #     translator = comp.varargs(full_name)
    # else:
    arity = len(op.__signature__.parameters)

    # TODO: Find the right way to do this...
    from sqlalchemy.sql.functions import _FunctionGenerator
    f = _FunctionGenerator()
    f._FunctionGenerator__names = [f'{db}', f'{func_name}']

    translator = fixed_arity(f, arity)

    SingleStoreExprTranslator._registry[op] = translator


def parse_type(t: str) -> Union[dt.DataType, Exception]:
    """
    Convert a string to an Ibis data type.

    Parameters
    ----------
    t : str
        Name of the data type

    Returns
    -------
    dt.DataType : if valid data type was found
    Exception : if string is not a valid type

    """
    t = t.lower()
    if t in _singlestore_to_ibis_type:
        return _singlestore_to_ibis_type[t]
    else:
        if 'varchar' in t or 'char' in t or 'text' in t:
            return 'string'
        elif 'decimal' in t:
            result = dt.dtype(t)
            if result:
                return t
            else:
                return ValueError(t)
        else:
            raise Exception(t)


_VARCHAR_RE = re.compile(r'varchar\((\d+)\)')


def _parse_varchar(t: str) -> Optional[str]:
    """
    Determine if given string is a varchar type name.

    Parameters
    ----------
    t : str
        Name of the data type

    Returns
    -------
    str : if input string contains a varchar name
    None : otherwise

    """
    m = _VARCHAR_RE.match(t)
    if m:
        return 'string'
    return None


def _singlestore_type_to_ibis(tval: str) -> str:
    """
    Convert SingleStore type to Ibis type name.

    Parameters
    ----------
    tval : str
        SingleStore data type name

    Returns
    -------
    str

    """
    if tval in _singlestore_to_ibis_type:
        return _singlestore_to_ibis_type[tval]
    return tval


def _ibis_string_to_singlestore(tval: str) -> Optional[str]:
    """
    Convert Ibis type name to Ibis type name.

    Parameters
    ----------
    tval : str
        Ibis data type name

    Returns
    -------
    str : if corresponding name exists
    None : otherwise

    """
    if tval in sql_type_names:
        return sql_type_names[tval]
    result = dt.validate_type(tval)
    if result:
        return repr(result)
    return None


_singlestore_to_ibis_type: Dict[str, str] = {
    'boolean': 'boolean',
    'tinyint': 'int8',
    'smallint': 'int16',
    'int': 'int32',
    'bigint': 'int64',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'text': 'string',
    'varchar': 'string',
    'char': 'string',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
}