#!/usr/bin/env python
"""SingleStoreDB function registry."""
from __future__ import annotations

import operator
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import ibis
import ibis.backends.base.sql.compiler.translator as tr
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.types.groupby as ig
import numpy as np
import pandas as pd
import sqlalchemy as sa
from ibis.backends.base.sql.alchemy import fixed_arity
from ibis.backends.base.sql.alchemy import sqlalchemy_operation_registry
from ibis.backends.base.sql.alchemy import sqlalchemy_window_functions_registry
from ibis.backends.base.sql.alchemy import unary

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)


def _substr(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    f = sa.func.substr

    arg, start, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _capitalize(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    (arg,) = expr.op().args
    sa_arg = t.translate(arg)
    return sa.func.concat(
        sa.func.ucase(sa.func.left(sa_arg, 1)), sa.func.substring(sa_arg, 2),
    )


def _extract(fmt: str) -> Callable[[tr.ExprTranslator, ir.Expr], ir.Expr]:
    def translator(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
        (arg,) = expr.op().args
        sa_arg = t.translate(arg)
        if fmt == 'millisecond':
            return sa.extract('microsecond', sa_arg) % 1000
        return sa.extract(fmt, sa_arg)

    return translator


_truncate_formats = {
    's': '%Y-%m-%d %H:%i:%s',
    'm': '%Y-%m-%d %H:%i:00',
    'h': '%Y-%m-%d %H:00:00',
    'D': '%Y-%m-%d',
    # 'W': 'week',
    'M': '%Y-%m-01',
    'Y': '%Y-01-01',
}


def _truncate(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, unit = expr.op().args
    sa_arg = t.translate(arg)
    try:
        fmt = _truncate_formats[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'Unsupported truncate unit {unit}',
        )
    return sa.func.date_format(sa_arg, fmt)


def _cast(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, typ = expr.op().args

    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(typ)

    # specialize going from an integer type to a timestamp
    if isinstance(arg.type(), dt.Integer) and isinstance(sa_type, sa.DateTime):
        return sa.func.convert_tz(sa.func.from_unixtime(sa_arg), 'SYSTEM', 'UTC')

    if arg.type().equals(dt.binary) and typ.equals(dt.string):
        return sa.func.hex(sa_arg)

    if typ.equals(dt.binary):
        #  decode yields a column of memoryview which is annoying to deal with
        # in pandas. CAST(expr AS BYTEA) is correct and returns byte strings.
        return sa.cast(sa_arg, sa.LargeBinary())

    return sa.cast(sa_arg, sa_type)


def _log(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    sa_base = t.translate(base)
    return sa.func.log(sa_base, sa_arg)


def _identical_to(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    left, right = args = expr.op().args
    if left.equals(right):
        return True
    else:
        left, right = map(t.translate, args)
        return left.op('<=>')(right)


def _round(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        sa_digits = 0
    else:
        sa_digits = t.translate(digits)

    return sa.func.round(sa_arg, sa_digits)


def _quantile(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, quantile, interpolation = expr.op().args
    sa_arg = t.translate(arg)
    sa_quantile = t.translate(quantile)
    return sa.func.approx_percentile(sa_arg, sa_quantile)


def _floor_divide(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    left, right = map(t.translate, expr.op().args)
    return sa.func.floor(left / right)


def _string_join(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    sep, elements = expr.op().args
    return sa.func.concat_ws(t.translate(sep), *map(t.translate, elements))


def _interval_from_integer(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    arg, unit = expr.op().args
    if unit in {'ms', 'ns'}:
        raise com.UnsupportedOperationError(
            'SingleStoreDB does not allow operation '
            'with INTERVAL offset {}'.format(unit),
        )

    sa_arg = t.translate(arg)
    text_unit = expr.type().resolution.upper()

    # XXX: Is there a better way to handle this? I.e. can we somehow use
    # the existing bind parameter produced by translate and reuse its name in
    # the string passed to sa.text?
    if isinstance(sa_arg, sa.sql.elements.BindParameter):
        return sa.text(f'INTERVAL :arg {text_unit}').bindparams(
            arg=sa_arg.value,
        )
    return sa.text(f'INTERVAL {sa_arg} {text_unit}')


def _timestamp_diff(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    left, right = expr.op().args
    sa_left = t.translate(left)
    sa_right = t.translate(right)
    return sa.func.timestampdiff(sa.text('SECOND'), sa_right, sa_left)


def _literal(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    if isinstance(expr, ir.IntervalScalar):
        if expr.type().unit in {'ms', 'ns'}:
            raise com.UnsupportedOperationError(
                'SingleStoreDB does not allow operation '
                'with INTERVAL offset {}'.format(expr.type().unit),
            )
        text_unit = expr.type().resolution.upper()
        value = expr.op().value
        return sa.text(f'INTERVAL :value {text_unit}').bindparams(value=value)
    elif isinstance(expr, ir.SetScalar):
        return list(map(sa.literal, expr.op().value))
    else:
        value = expr.op().value
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        return sa.literal(value)


def _random(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    return sa.func.random()


def _group_concat(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    op = expr.op()
    arg, sep, where = op.args
    if where is not None:
        case = where.ifelse(arg, ibis.NA)
        arg = t.translate(case)
    else:
        arg = t.translate(arg)
    return sa.func.group_concat(arg.op('SEPARATOR')(t.translate(sep)))


def _day_of_week_index(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    (arg,) = expr.op().args
    left = sa.func.dayofweek(t.translate(arg)) - 2
    right = 7
    return ((left % right) + right) % right


def _day_of_week_name(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    (arg,) = expr.op().args
    return sa.func.dayname(t.translate(arg))


def _string_find(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    op = expr.op()

    if op.end is not None:
        raise NotImplementedError('`end` not yet implemented')

    if op.start is not None:
        return sa.func.locate(
            t.translate(op.substr),
            t.translate(op.arg),
            t.translate(op.start),
        ) - 1

    return sa.func.locate(t.translate(op.substr), t.translate(op.arg)) - 1


def _string_contains(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    return _string_find(t, expr) >= 0


def _approx_median(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    args = expr.op().args
    return sa.func.median(t.translate(args[0]))


def _regex_search(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    args = expr.op().args
    return sa.func.regexp_instr(*[t.translate(x) for x in args], sa.literal('g')) > 0


def _regex_replace(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    # TODO: Requires regexp_format='advanced'
    args = expr.op().args
    return sa.func.regexp_replace(*[t.translate(x) for x in args], sa.literal('g'))


def _regex_extract(t: tr.ExprTranslator, expr: ir.Expr) -> ir.Expr:
    args = expr.op().args
    return sa.func.regexp_extract(*[t.translate(x) for x in args], sa.literal('g'))


operation_registry.update(
    {
        ops.Literal: _literal,
        # strings
        ops.Substring: _substr,
        ops.StringFind: _string_find,
        ops.StringContains: _string_contains,
        ops.Capitalize: _capitalize,
        ops.RegexSearch: _regex_search,
        ops.RegexReplace: _regex_replace,
        ops.RegexExtract: _regex_extract,
        ops.Cast: _cast,
        # math
        ops.Log: _log,
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.Round: _round,
        ops.RandomScalar: _random,
        ops.Quantile: _quantile,
        # dates and times
        ops.Date: unary(sa.func.date),
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(sa.func.datediff, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: _timestamp_diff,
        ops.DateTruncate: _truncate,
        ops.TimestampTruncate: _truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.ExtractYear: _extract('year'),
        ops.ExtractMonth: _extract('month'),
        ops.ExtractDay: _extract('day'),
        ops.ExtractDayOfYear: unary('dayofyear'),
        ops.ExtractQuarter: _extract('quarter'),
        ops.ExtractEpochSeconds: unary('UNIX_TIMESTAMP'),
        ops.ExtractWeekOfYear: fixed_arity('weekofyear', 1),
        ops.ExtractHour: _extract('hour'),
        ops.ExtractMinute: _extract('minute'),
        ops.ExtractSecond: _extract('second'),
        ops.ExtractMillisecond: _extract('millisecond'),
        # reductions
        ops.ApproxMedian: _approx_median,
        #       ops.BitAnd: reduction(sa.func.bit_and),
        #       ops.BitOr: reduction(sa.func.bit_or),
        #       ops.BitXor: reduction(sa.func.bit_xor),
        #       ops.Variance: variance_reduction('var'),
        #       ops.StandardDev: variance_reduction('stddev'),
        #       ops.IdenticalTo: _identical_to,
        #       ops.TimestampNow: fixed_arity(sa.func.now, 0),
        # others
        ops.GroupConcat: _group_concat,
        ops.DayOfWeekIndex: _day_of_week_index,
        ops.DayOfWeekName: _day_of_week_name,
        #       ops.HLLCardinality: reduction(
        #           lambda arg: sa.func.count(arg.distinct()),
        #       ),
    },
)


def _describe_table(
    self: Any,
    percentiles: Sequence[float] = [0.25, 0.5, 0.75],
    include: Optional[Sequence[Any]] = None,
    exclude: Optional[Sequence[Any]] = None,
    datetime_is_numeric: bool = False,
    stats: Optional[Sequence[str]] = None,
    by: Optional[Sequence[str | ir.Expr]] = None,
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
        np.object: 'object', np.datetime64: 'datetime',
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
        elif isinstance(dtype, (dt.String, dt.Binary, dt.Set, dt.Enum)):
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


ig.GroupedTable.describe = _grouped_describe


def _corr(
    self: Any,
    index_name: Optional[str] = 'Variable',
    diagonals: Union[float, int] = 1,
    by: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Compute the correlation matrix for all numeric variables.

    Parameters
    ----------
    index_name : str, optional
        The name of the column containing the variable names
    diagonals : float or int, optional
        Values of the diagnol components
    by : str or list-of-str, optional
        Group by variables

    Returns
    -------
    DataFrame

    """
    if isinstance(self, ig.GroupedTable):
        table = self.table
    else:
        table = self

    num_vars = []
    for name, dtype in table.schema().items():
        if isinstance(dtype, (dt.Integer, dt.Floating, dt.Decimal)):
            num_vars.append(name)

    from sqlalchemy_singlestoredb import array

    group_vars = []
    if by is not None:
        if isinstance(by, str):
            group_vars = [by]
        else:
            group_vars = by

    query = sa.select(
        *[sa.column(x) for x in group_vars], sa.func.corrmat(
            sa.func.vec_pack_f64(
                array(
                    *[sa.column(x) for x in num_vars],
                ),
            ),
        ),
    ).select_from(table.compile().subquery())

    if by is not None and by:
        query = query.group_by(*[sa.column(x) for x in by])

    n = len(num_vars)
    out = []
    df = table.sql(str(query)).execute()
    for row in df.itertuples(index=False):
        index = list(row[:-1])
        corr = row[-1]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1):
                mat[i, j] = corr.pop(0)
        mat = mat + np.transpose(mat)
        for i in range(n):
            mat[i, i] = diagonals

        for i, item in enumerate(index):
            index[i] = [item] * n

        index.append(num_vars)

        out.append(pd.DataFrame(mat, columns=num_vars, index=index))
        out[-1].index.names = group_vars + [index_name]  # type: ignore

    return pd.concat(out)


ir.Table.corr = _corr


def _grouped_corr(
    self: Any,
    index_name: Optional[str] = 'Variable',
    diagonals: Union[float, int] = 1,
) -> pd.DataFrame:
    """
    Compute the correlation matrix for all numeric variables.

    Parameters
    ----------
    index_name : str, optional
        The name of the column containing the variable names
    diagonals : float or int, optional
        Values of the diagnol components

    Returns
    -------
    DataFrame

    """
    return _corr(
        self, by=[x.get_name() for x in self.by],
        index_name=index_name, diagonals=diagonals,
    )


ig.GroupedTable.corr = _grouped_corr


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


def _distinct_column(self: Any) -> ir.Expr:
    """Return distinct values."""
    return self.to_projection().distinct()[0]


ir.AnyColumn.distinct = _distinct_column


def _drop_duplicates(
    self: Any,
    subset: Optional[str | Sequence[str]] = None,
    keep: str = 'first',
    order_by: Optional[str | Sequence[str] | ir.Expr] = None,
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


def _shape(self: Any) -> Tuple[int, int]:
    """Return the shape of the table."""
    return self.count().execute(), len(self.columns)


ir.Table.shape = property(_shape)


def _dtypes(self: Any) -> pd.Series:
    """Return the data types of the columns."""
    items = list(self.schema.items())
    return pd.Series([x[1] for x in items], index=[x[0] for x in items])


ir.Table.dtypes = property(_dtypes)
