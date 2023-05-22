from __future__ import annotations

import contextlib
import functools
import operator
import string
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.types.groupby as gby
import numpy as np
import pandas as pd
import sqlalchemy as sa
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator as ExprTranslator
from ibis.backends.base.sql.alchemy import fixed_arity
from ibis.backends.base.sql.alchemy import sqlalchemy_operation_registry
from ibis.backends.base.sql.alchemy import sqlalchemy_window_functions_registry
from ibis.backends.base.sql.alchemy import unary
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.registry import geospatial_functions
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction


operation_registry = sqlalchemy_operation_registry.copy()

# NOTE: window functions are available from MySQL 8 and MariaDB 10.2
operation_registry.update(sqlalchemy_window_functions_registry)

if geospatial_supported:
    operation_registry.update(geospatial_functions)


def _substr(t: ExprTranslator, op: ops.Substring) -> ir.Expr:
    f = sa.func.substr

    sa_arg = t.translate(op.arg)
    sa_start = t.translate(op.start)

    if op.length is None:
        return f(sa_arg, sa_start + 1)

    sa_length = t.translate(op.length)
    return f(sa_arg, sa_start + 1, sa_length)


def _capitalize(t: ExprTranslator, op: ops.Capitalize) -> ir.Expr:
    sa_arg = t.translate(op.arg)
    return sa.func.concat(
        sa.func.ucase(sa.func.left(sa_arg, 1)), sa.func.substring(sa_arg, 2),
    )


_truncate_formats = {
    's': '%Y-%m-%d %H:%i:%s',
    'm': '%Y-%m-%d %H:%i:00',
    'h': '%Y-%m-%d %H:00:00',
    'D': '%Y-%m-%d',
    # 'W': 'week',
    'M': '%Y-%m-01',
    'Y': '%Y-01-01',
}


def _truncate(
    t: ExprTranslator,
    op: Union[ops.TimestampTruncate, ops.DateTruncate, ops.TimeTruncate],
) -> ir.Expr:
    sa_arg = t.translate(op.arg)
    try:
        fmt = _truncate_formats[op.unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(f'Unsupported truncate unit {op.unit}')
    return sa.func.date_format(sa_arg, fmt)


def _cast(t: ExprTranslator, op: ops.Cast) -> ir.Expr:
    arg = op.arg
    to = op.to
    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(op.to)

    # specialize going from an integer type to a timestamp
    if isinstance(arg.output_dtype, dt.Integer) and isinstance(sa_type, sa.DateTime):
        return sa.func.convert_tz(sa.func.from_unixtime(sa_arg), 'SYSTEM', 'UTC')

    if arg.output_dtype.equals(dt.binary) and to.equals(dt.string):
        return sa.func.hex(sa_arg)

    if to.equals(dt.binary):
        #  decode yields a column of memoryview which is annoying to deal with
        # in pandas. CAST(expr AS BYTEA) is correct and returns byte strings.
        return sa.cast(sa_arg, sa.LargeBinary())

    return sa.cast(sa_arg, sa_type)


def _round(t: ExprTranslator, op: ops.Round) -> ir.Expr:
    sa_arg = t.translate(op.arg)
    if op.digits is None:
        sa_digits = 0
    else:
        sa_digits = t.translate(op.digits)
    return sa.func.round(sa_arg, sa_digits)


def _quantile(t: ExprTranslator, op: ops.Quantile) -> ir.Expr:
    sa_arg = t.translate(op.arg)
    sa_quantile = t.translate(op.quantile)
    return sa.func.approx_percentile(sa_arg, sa_quantile)


def _interval_from_integer(t: ExprTranslator, op: ops.IntervalFromInteger) -> ir.Expr:
    if op.unit.short in {'ms', 'ns'}:
        raise com.UnsupportedOperationError(
            f'SingloStoreDB does not allow operation with INTERVAL offset {op.unit}',
        )

    sa_arg = t.translate(op.arg)
    text_unit = op.output_dtype.resolution.upper()

    # XXX: Is there a better way to handle this? I.e. can we somehow use
    # the existing bind parameter produced by translate and reuse its name in
    # the string passed to sa.text?
    if isinstance(sa_arg, sa.sql.elements.BindParameter):
        return sa.text(f'INTERVAL :arg {text_unit}').bindparams(arg=sa_arg.value)
    return sa.text(f'INTERVAL {sa_arg} {text_unit}')


def _literal(t: ExprTranslator, op: ops.Literal) -> ir.Expr:
    if op.output_dtype.is_interval():
        if op.output_dtype.unit.short in {'ms', 'ns'}:
            raise com.UnsupportedOperationError(
                'SingloStoreDB does not allow operation '
                f'with INTERVAL offset {op.output_dtype.unit}',
            )
        text_unit = op.output_dtype.resolution.upper()
        sa_text = sa.text(f'INTERVAL :value {text_unit}')
        return sa_text.bindparams(value=op.value)

    elif op.output_dtype.is_binary():
        # the cast to BINARY is necessary here, otherwise the data come back as
        # Python strings
        #
        # This lets the database handle encoding rather than ibis
        return sa.cast(sa.literal(op.value), type_=sa.BINARY())

    value = op.value
    with contextlib.suppress(AttributeError):
        value = value.to_pydatetime()

    return sa.literal(value)


def _group_concat(t: ExprTranslator, op: ops.GroupConcat) -> ir.Expr:
    if op.where is not None:
        arg = t.translate(ops.Where(op.where, op.arg, ibis.NA))
    else:
        arg = t.translate(op.arg)
    sep = t.translate(op.sep)
    return sa.func.group_concat(arg.op('SEPARATOR')(sep))


def _json_get_item(t: ExprTranslator, op: ops.JSONGetItem) -> ir.Expr:
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    if op.index.output_dtype.is_integer():
        path = '$[' + sa.cast(index, sa.TEXT) + ']'
    else:
        path = '$.' + index
    return sa.func.json_extract(arg, path)


class _singlestoredb_trim(GenericFunction):
    inherit_cache = True

    def __init__(self, input: ir.Expr, side: str) -> None:
        super().__init__(input)
        self.type = sa.VARCHAR()
        self.side = side


@compiles(_singlestoredb_trim, 'singlestoredb')
def compiles_singlestoredb_trim(
    element: ir.Expr,
    compiler: Any,
    **kw: Dict[str, Any],
) -> ir.Expr:
    arg = compiler.function_argspec(element, **kw)
    side = element.side.upper()
    # has to be called once for every whitespace character because singlestoredb
    # interprets `char` literally, not as a set of characters like Python
    return functools.reduce(
        lambda arg, char: f"TRIM({side} '{char}' FROM {arg})", string.whitespace, arg,
    )


def _string_find(t: ExprTranslator, op: ops.StringFind) -> ir.Expr:
    if op.end is not None:
        raise NotImplementedError('`end` not yet implemented')

    if op.start is not None:
        return sa.func.locate(
            t.translate(op.substr),
            t.translate(op.arg),
            t.translate(op.start),
        ) - 1

    return sa.func.locate(t.translate(op.substr), t.translate(op.arg)) - 1


def _string_contains(t: ExprTranslator, op: ops.StringContains) -> ir.Expr:
    return (sa.func.locate(t.translate(op.needle), t.translate(op.haystack)) - 1) >= 0


def _approx_median(t: ExprTranslator, op: ops.ApproxMedian) -> ir.Expr:
    return sa.func.median(t.translate(op.arg))


def _regex_search(t: ExprTranslator, op: ops.RegexSearch) -> ir.Expr:
    args = (op.arg, op.pattern)
    return sa.type_coerce(
        sa.func.regexp_instr(*[t.translate(x) for x in args], sa.literal('g')) > 0,
        sa.BOOLEAN,
    )


def _regex_replace(t: ExprTranslator, op: ops.RegexReplace) -> ir.Expr:
    # TODO: Requires regexp_format='advanced'
    args = (op.arg, op.pattern, op.replacement)
    return sa.func.regexp_replace(*[t.translate(x) for x in args], sa.literal('g'))


def _regex_extract(t: ExprTranslator, op: ops.RegexExtract) -> ir.Expr:
    args = (op.arg, op.pattern, op.index)
    return sa.func.regexp_extract(*[t.translate(x) for x in args], sa.literal('g'))


operation_registry.update(
    {
        ops.Literal: _literal,
        ops.Cast: _cast,
        ops.IfNull: fixed_arity(sa.func.ifnull, 2),
        # static checks are not happy with using "if" as a property
        ops.Where: fixed_arity(getattr(sa.func, 'if'), 3),
        # strings
        ops.Substring: _substr,
        ops.StringFind: _string_find,
        ops.StringContains: _string_contains,
        ops.Capitalize: _capitalize,
        ops.FindInSet: (
            lambda t, op: (
                sa.func.find_in_set(
                    t.translate(op.needle),
                    sa.func.concat_ws(',', *map(t.translate, op.values)),
                )
                - 1
            )
        ),
        # LIKE in singlestoredb is case insensitive
        ops.StartsWith: fixed_arity(
            lambda arg, start: sa.type_coerce(
                arg.op('LIKE BINARY')(sa.func.concat(start, '%')), sa.BOOLEAN(),
            ),
            2,
        ),
        ops.EndsWith: fixed_arity(
            lambda arg, end: sa.type_coerce(
                arg.op('LIKE BINARY')(sa.func.concat('%', end)), sa.BOOLEAN(),
            ),
            2,
        ),
        ops.RegexSearch: _regex_search,
        ops.RegexReplace: _regex_replace,
        ops.RegexExtract: _regex_extract,
        # math
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.Round: _round,
        ops.Quantile: _quantile,
        # dates and times
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(sa.func.datediff, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: fixed_arity(
            lambda left, right: sa.func.timestampdiff(sa.text('SECOND'), right, left), 2,
        ),
        ops.StringToTimestamp: fixed_arity(
            lambda arg, format_str: sa.func.str_to_date(arg, format_str), 2,
        ),
        ops.DateTruncate: _truncate,
        ops.TimestampTruncate: _truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.ExtractDayOfYear: unary(sa.func.dayofyear),
        ops.ExtractEpochSeconds: unary(sa.func.UNIX_TIMESTAMP),
        ops.ExtractWeekOfYear: unary(sa.func.weekofyear),
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.func.floor(sa.extract('microsecond', arg) / 1000), 1,
        ),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        # reductions
        ops.ApproxMedian: _approx_median,
        # others
        ops.GroupConcat: _group_concat,
        ops.DayOfWeekIndex: fixed_arity(
            lambda arg: (sa.func.dayofweek(arg) + 5) % 7, 1,
        ),
        ops.DayOfWeekName: fixed_arity(lambda arg: sa.func.dayname(arg), 1),
        ops.JSONGetItem: _json_get_item,
        ops.Strip: unary(lambda arg: _singlestoredb_trim(arg, 'both')),
        ops.LStrip: unary(lambda arg: _singlestoredb_trim(arg, 'leading')),
        ops.RStrip: unary(lambda arg: _singlestoredb_trim(arg, 'trailing')),
    },
)

_invalid_operations = {
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeMax,
    ops.CumulativeMean,
    ops.CumulativeMin,
    ops.CumulativeSum,
    ops.NTile,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}


def _describe_table(
    self: Any,
    percentiles: Sequence[float] = [0.25, 0.5, 0.75],
    include: Optional[Sequence[Any]] = None,
    exclude: Optional[Sequence[Any]] = None,
    datetime_is_numeric: bool = False,
    stats: Optional[Sequence[str]] = None,
    by: Optional[Sequence[Union[str, ir.Expr]]] = None,
    having: Optional[Sequence[ir.Expr]] = None,
) -> ir.TableExpr:
    """
    Compute descriptive statistics.

    Note that since all statistics are stored in columns, there can
    be mixed data types represented in some columns such as ``min``,
    ``max``, ``top``, etc. Because of this, any column that contains
    the actual value of a column has a string type and the value
    must be parsed from the return value.

    Parameters
    ----------
    percentiles : list-like of floats, optional
        The percentiles to compute. Values should be between 0 and 1.
    include : 'all' or list-like of dtypes, optional
        By default, only numeric columns are included in the results.
        Specify 'all' to include all variables. A list of data types
        can also be specified. For example, ``numpy.number`` will
        select all numerics, ``numpy.object`` will select all
        objects types (i.e., strings and binary).
    exclude : list-like of dtypes, optional
        A list of data types to exclude.
    datetime_is_numeric : bool, optional
        Whether to treat datetimes as numeric. This affects the statistics
        computed for the column.
    stats : list-like of strings, optional
        Statistics to include in the output. By default, statistics are
        chosen based on data type. For numerics, ``count``, ``mean``,
        ``median``, ``std``, ``var``, ``min``, ``pct``, and ``max``
        are computed. For strings and binary, ``count``, ``unique``,
        ``min``, ``max``, ``top``, and ``freq`` are computed. For datetimes,
        ``count``, ``unique``, ``min``, ``max``, ``top``, and ``freq``
        are computed. If ``datetime_is_numeric`` is specified, all numeric
        computations are done for datetimes as well.
    by : list-like of strings or expressions, optional
        Group by variables
    having : list-like of expressions, optional
        Expressions that filter based on aggregate value

    Returns
    -------
    TableExpr

    """
    stat_map = dict(unique='nunique', median='approx_median')
    type_map = dict(
        std='double', mean='double', var='double',
        count='int64', unique='int64',
    )

    # Valid stats for each type
    num_stats = set('count mean median std var min pct max'.split())
    str_stats = set('count unique min max top freq'.split())
    dt_stats = set('count unique min max top freq'.split())

    if datetime_is_numeric:
        dt_stats.union(num_stats)

    # Compute includes and excludes

    if include is None:
        include = ['number']
    elif not isinstance(include, Sequence) or isinstance(include, str):
        include = [include]

    if exclude is None:
        exclude = []
    elif not isinstance(exclude, Sequence) or isinstance(exclude, str):
        exclude = [exclude]

    include_map = {
        np.datetime64: 'datetime',
        'datetime64': 'datetime', np.timedelta64: 'timedelta',
        'timedelta64': 'timedelta', 'category': 'object',
        'datetimetz': 'datetime', 'datetime64[ns, tz]': 'datetime',
        int: 'int', 'i': 'int', float: 'float', 'f': 'float',
        str: 'string', 'str': 'string', bytes: 'binary',
        'bytes': 'binary', 'O': 'object',
    }

    include_set = set([include_map.get(x, x) for x in include])
    exclude_set = set([include_map.get(x, x) for x in exclude])
    include_set = include_set.difference(exclude_set)

    include_vars = []
    for name, dtype in self.schema().items():
        if 'all' in include_set or 'one' in include_set:
            include_vars.append((name, dtype))
        elif 'object' in include_set and \
                isinstance(dtype, (dt.String, dt.Binary)):
            include_vars.append((name, dtype))
        elif 'string' in include_set and \
                isinstance(dtype, dt.String):
            include_vars.append((name, dtype))
        elif 'number' in include_set and \
                isinstance(dtype, (dt.Integer, dt.Floating, dt.Decimal)):
            include_vars.append((name, dtype))
        elif 'number' in include_set and datetime_is_numeric and \
                isinstance(dtype, (dt.Timestamp, dt.Date, dt.Time, dt.Interval)):
            include_vars.append((name, dtype))
        elif 'float' in include_set and \
                isinstance(dtype, (dt.Floating, dt.Decimal)):
            include_vars.append((name, dtype))
        elif 'int' in include_set and \
                isinstance(dtype, dt.Integer):
            include_vars.append((name, dtype))
        elif 'datetime' in include_set and \
                isinstance(dtype, (dt.Timestamp, dt.Date, dt.Time)):
            include_vars.append((name, dtype))
        elif 'timedelta' in include_set and \
                isinstance(dtype, dt.Interval):
            include_vars.append((name, dtype))
        elif 'bytes' in include_set and \
                isinstance(dtype, dt.Binary):
            include_vars.append((name, dtype))

    if not include_vars:
        raise ValueError(
            'No variables selected. Use `include=` or `exclude=` '
            'to select valid data types.',
        )

    # Compute stats list
    type_stats: Set[str] = set()
    for name, dtype in include_vars:
        if isinstance(dtype, (dt.Timestamp, dt.Date, dt.Time, dt.Interval)):
            type_stats = type_stats.union(dt_stats)
        elif isinstance(dtype, (dt.String, dt.Binary, dt.Array, dt.Enum)):
            type_stats = type_stats.union(str_stats)
        else:
            type_stats = type_stats.union(num_stats)

    stats = [
        x for x in [
            'count', 'unique', 'mean', 'median', 'std', 'var',
            'min', 'pct', 'max', 'top', 'freq',
        ]
        if x in type_stats and (stats is None or x in stats)
    ]

    union = []

    # Build table expression
    if 'one' in include_set:
        out_type = str(include_vars[0][-1])
    else:
        out_type = 'string'
    for name, dtype in include_vars:
        if 'one' in include_set:
            agg = {}
        else:
            agg = {'name': ibis.literal(name)}
        freq = None
        for stat in stats:
            mthd_name = stat_map.get(stat, stat)
            # Expand 'pct' to percentile values
            if stat == 'pct':
                for pct in percentiles:
                    pct_label = str(int(pct * 100)) + '%'
                    if hasattr(self[name], 'quantile'):
                        agg[pct_label] = self[name].quantile(pct).cast(out_type)
                    else:
                        agg[pct_label] = ibis.null().cast(out_type)
            elif stat in ['top', 'freq']:
                freq = self[name].topk(1).to_aggregation(metric_name='freq')[
                    lambda x: x[name].cast(out_type).name('top'), 'freq',
                ]
            else:
                mthd = getattr(self[name], mthd_name, None)
                if mthd is None:
                    agg[stat] = ibis.null().cast(type_map.get(stat, out_type))
                elif stat == 'median' and isinstance(dtype, dt.String):
                    agg[stat] = ibis.null().cast(type_map.get(stat, out_type))
                else:
                    agg[stat] = mthd().cast(type_map.get(stat, out_type))

        union.append(self.aggregate(**agg, by=by, having=having))

        if freq is not None:
            union[-1] = union[-1].cross_join(freq)

    return ibis.union(*union)


ir.Table.describe = _describe_table


def _grouped_describe(self: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Compute descriptive statistics.

    Note that since all statistics are stored in columns, there can
    be mixed data types represented in some columns such as ``min``,
    ``max``, ``top``, etc. Because of this, any column that contains
    the actual value of a column has a string type and the value
    must be parsed from the return value.

    Parameters
    ----------
    percentiles : list-like of floats, optional
        The percentiles to compute. Values should be between 0 and 1.
    include : 'all' or list-like of dtypes, optional
        By default, only numeric columns are included in the results.
        Specify 'all' to include all variables. A list of data types
        can also be specified. For example, ``numpy.number`` will
        select all numerics, ``numpy.object`` will select all
        objects types (i.e., strings and binary).
    exclude : list-like of dtypes, optional
        A list of data types to exclude.
    datetime_is_numeric : bool, optional
        Whether to treat datetimes as numeric. This affects the statistics
        computed for the column.
    stats : list-like of strings, optional
        Statistics to include in the output. By default, statistics are
        chosen based on data type. For numerics, ``count``, ``mean``,
        ``median``, ``std``, ``var``, ``min``, ``pct``, and ``max``
        are computed. For strings and binary, ``count``, ``unique``,
        ``min``, ``max``, ``top``, and ``freq`` are computed. For datetimes,
        ``count``, ``unique``, ``min``, ``max``, ``top``, and ``freq``
        are computed. If ``datetime_is_numeric`` is specified, all numeric
        computations are done for datetimes as well.

    Returns
    -------
    TableExpr

    """
    return self.table.describe(*args, by=self.by, having=self._having, **kwargs)\
                     .select(lambda x: x).sort_by([x.get_name() for x in self.by])


gby.GroupedTable.describe = _grouped_describe


def _describe_column(
    self: Any,
    percentiles: Sequence[float] = [0.25, 0.5, 0.75],
    datetime_is_numeric: bool = False,
    stats: Optional[Sequence[str]] = None,
) -> ir.TableExpr:
    """
    Compute descriptive statistics.

    Parameters
    ----------
    percentiles : list-like of floats, optional
        The percentiles to compute. Values should be between 0 and 1.
    datetime_is_numeric : bool, optional
        Whether to treat datetimes as numeric. This affects the statistics
        computed for the column.
    stats : list-like of strings, optional
        Statistics to include in the output. By default, statistics are
        chosen based on data type. For numerics, ``count``, ``mean``,
        ``median``, ``std``, ``var``, ``min``, ``pct``, and ``max``
        are computed. For strings and binary, ``count``, ``unique``,
        ``min``, ``max``, ``top``, and ``freq`` are computed. For datetimes,
        ``count``, ``unique``, ``min``, ``max``, ``top``, and ``freq``
        are computed. If ``datetime_is_numeric`` is specified, all numeric
        computations are done for datetimes as well.

    Returns
    -------
    TableExpr

    """
    return _describe_table(
        self.to_projection(), percentiles=percentiles,
        include='one', exclude=None, datetime_is_numeric=datetime_is_numeric, stats=stats,
    )


ir.AnyColumn.describe = _describe_column


def _head_column(self: ir.AnyColumn, n: int = 5) -> ir.Expr:
    """Return first ``n`` row values."""
    return self.to_projection().head(n)[0]


ir.AnyColumn.head = _head_column


def _drop_duplicates(
    self: Any,
    subset: Optional[Union[str, Sequence[str]]] = None,
    keep: str = 'first',
    order_by: Optional[Union[str, Sequence[str], ir.Expr]] = None,
) -> ir.Table:
    """
    Drop rows with duplicate values.

    Parameters
    ----------
    subset : str or list-of-strs, optional
        The name or names of columns to compare for duplicate values
    keep : str, optional
        Which duplicate to keep: first or last
    order_by : str or Expr, optional
        The sort order for the duplicates to determine first or last

    Returns
    -------
    Table

    """
    if keep not in ['first', 'last']:
        raise ValueError('`keep` must be "first" or "last"')

    if order_by is None:
        raise ValueError(
            '`order_by` must be specified for '
            '`keep="first"` or `keep="last"`',
        )

    if subset is None:
        subset = list(self.columns)
    elif isinstance(subset, str):
        subset = [subset]

    # TODO: Workaround for bug in Ibis where ibis.desc/ibis.asc can't
    #       be used in an order_by.
    def order_func(obj: Any) -> Any:
        func = ibis.asc if keep == 'first' else ibis.desc
        return func(order_by).resolve(obj)

    return self.group_by(subset).order_by(order_func)\
        .mutate(**{'^ROW^ORDER^': ibis.row_number()})\
        .filter(lambda x: x['^ROW^ORDER^'] == 0).drop('^ROW^ORDER^')


ir.Table.drop_duplicates = _drop_duplicates
