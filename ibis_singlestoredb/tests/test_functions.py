from __future__ import annotations

from typing import Any

import ibis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

ROW_IDS = [21, 25, 31]
FLOAT_VECTORS = [
    [0.1, 2.5, -6.7],
    [10.3, -3.3, 7.8],
    [-2.3, 6.6, 7.9],
]
FLOAT32_SERIES = pd.Series(FLOAT_VECTORS, dtype='<f4')
FLOAT64_SERIES = pd.Series(FLOAT_VECTORS, dtype='<f8')
FLOAT32_VECTOR_BYTES = [np.array(x, dtype='<f4').tobytes() for x in FLOAT_VECTORS]
FLOAT64_VECTOR_BYTES = [np.array(x, dtype='<f8').tobytes() for x in FLOAT_VECTORS]
INT_VECTORS = [
    [0, 3, -7],
    [10, -3, 8],
    [-2, 7, 8],
]
INT8_SERIES = pd.Series(INT_VECTORS, dtype='<i1')
INT16_SERIES = pd.Series(INT_VECTORS, dtype='<i2')
INT32_SERIES = pd.Series(INT_VECTORS, dtype='<i4')
INT64_SERIES = pd.Series(INT_VECTORS, dtype='<i8')
INT8_VECTOR_BYTES = [np.array(x, dtype='<i1').tobytes() for x in INT_VECTORS]
INT16_VECTOR_BYTES = [np.array(x, dtype='<i2').tobytes() for x in INT_VECTORS]
INT32_VECTOR_BYTES = [np.array(x, dtype='<i4').tobytes() for x in INT_VECTORS]
INT64_VECTOR_BYTES = [np.array(x, dtype='<i8').tobytes() for x in INT_VECTORS]
LISTS = [
    [0.1, 'foo', True],
    [10.3, 'bar', False],
    [0.3, 'baz', False],
]
OBJECTS = [
    {'a': 1, 'b': 2, 'c': {'d': 100}},
    {'a': 1, 'b': 12, 'c': {'d': 105, 'e': True}},
    {'a': 10, 'b': 25, 'c': {'d': 111, 'e': 1.234}},
]


def make_vectors(tbl: Any) -> Any:
    return tbl.mutate(
        vec1=lambda x: x.text_vector.json_array_pack(),
        vec2=lambda x: x.text_vector.json_array_pack(),
        vec1_f32=lambda x: x.text_vector.json_array_pack_f32(),
        vec2_f32=lambda x: x.text_vector.json_array_pack_f32(),
        vec1_f64=lambda x: x.text_vector.json_array_pack_f64(),
        vec2_f64=lambda x: x.text_vector.json_array_pack_f64(),
        vec1_i8=lambda x: x.text_vector.json_array_pack_i8(),
        vec2_i8=lambda x: x.text_vector.json_array_pack_i8(),
        vec1_i16=lambda x: x.text_vector.json_array_pack_i16(),
        vec2_i16=lambda x: x.text_vector.json_array_pack_i16(),
        vec1_i32=lambda x: x.text_vector.json_array_pack_i32(),
        vec2_i32=lambda x: x.text_vector.json_array_pack_i32(),
        vec1_i64=lambda x: x.text_vector.json_array_pack_i64(),
        vec2_i64=lambda x: x.text_vector.json_array_pack_i64(),
    )


def test_json_array_pack(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        # default (float32)
        text_packed=lambda x: x.text_vector.json_array_pack(),
        json_packed=lambda x: x.json_vector.json_array_pack(),
        text_unpacked=lambda x: x.text_vector.json_array_pack().json_array_unpack(),
        json_unpacked=lambda x: x.json_vector.json_array_pack().json_array_unpack(),

        # float32
        text_float32_packed=lambda x: x.text_vector.json_array_pack_f32(),
        json_float32_packed=lambda x: x.json_vector.json_array_pack_f32(),
        text_float32_unpacked=lambda x: x.text_vector.json_array_pack_f32()\
                                                     .json_array_unpack_f32(),
        json_float32_unpacked=lambda x: x.json_vector.json_array_pack_f32()\
                                                     .json_array_unpack_f32(),

        # float64
        text_float64_packed=lambda x: x.text_vector.json_array_pack_f64(),
        json_float64_packed=lambda x: x.json_vector.json_array_pack_f64(),
        text_float64_unpacked=lambda x: x.text_vector.json_array_pack_f64()\
                                                     .json_array_unpack_f64(),
        json_float64_unpacked=lambda x: x.json_vector.json_array_pack_f64()\
                                                     .json_array_unpack_f64(),

        # int8
        text_int8_packed=lambda x: x.text_vector.json_array_pack_i8(),
        json_int8_packed=lambda x: x.json_vector.json_array_pack_i8(),
        text_int8_unpacked=lambda x: x.text_vector.json_array_pack_i8()\
                                                  .json_array_unpack_i8(),
        json_int8_unpacked=lambda x: x.json_vector.json_array_pack_i8()\
                                                  .json_array_unpack_i8(),

        # int16
        text_int16_packed=lambda x: x.text_vector.json_array_pack_i16(),
        json_int16_packed=lambda x: x.json_vector.json_array_pack_i16(),
        text_int16_unpacked=lambda x: x.text_vector.json_array_pack_i16()\
                                                   .json_array_unpack_i16(),
        json_int16_unpacked=lambda x: x.json_vector.json_array_pack_i16()\
                                                   .json_array_unpack_i16(),

        # int32
        text_int32_packed=lambda x: x.text_vector.json_array_pack_i32(),
        json_int32_packed=lambda x: x.json_vector.json_array_pack_i32(),
        text_int32_unpacked=lambda x: x.text_vector.json_array_pack_i32()\
                                                   .json_array_unpack_i32(),
        json_int32_unpacked=lambda x: x.json_vector.json_array_pack_i32()\
                                                   .json_array_unpack_i32(),

        # int64
        text_int64_packed=lambda x: x.text_vector.json_array_pack_i64(),
        json_int64_packed=lambda x: x.json_vector.json_array_pack_i64(),
        text_int64_unpacked=lambda x: x.text_vector.json_array_pack_i64()\
                                                   .json_array_unpack_i64(),
        json_int64_unpacked=lambda x: x.json_vector.json_array_pack_i64()\
                                                   .json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_packed.tolist() == FLOAT32_VECTOR_BYTES
    assert out.json_packed.tolist() == FLOAT32_VECTOR_BYTES
    tm.assert_series_equal(out.text_unpacked, FLOAT32_SERIES, check_names=False)
    tm.assert_series_equal(out.json_unpacked, FLOAT32_SERIES, check_names=False)

    assert out.text_float32_packed.tolist() == FLOAT32_VECTOR_BYTES
    assert out.json_float32_packed.tolist() == FLOAT32_VECTOR_BYTES
    tm.assert_series_equal(out.text_float32_unpacked, FLOAT32_SERIES, check_names=False)
    tm.assert_series_equal(out.json_float32_unpacked, FLOAT32_SERIES, check_names=False)

    assert out.text_float64_packed.tolist() == FLOAT64_VECTOR_BYTES
    assert out.json_float64_packed.tolist() == FLOAT64_VECTOR_BYTES
    tm.assert_series_equal(out.text_float64_unpacked, FLOAT64_SERIES, check_names=False)
    tm.assert_series_equal(out.json_float64_unpacked, FLOAT64_SERIES, check_names=False)

    assert out.text_int8_packed.tolist() == INT8_VECTOR_BYTES
    assert out.json_int8_packed.tolist() == INT8_VECTOR_BYTES
    tm.assert_series_equal(out.text_int8_unpacked, INT8_SERIES, check_names=False)
    tm.assert_series_equal(out.json_int8_unpacked, INT8_SERIES, check_names=False)

    assert out.text_int16_packed.tolist() == INT16_VECTOR_BYTES
    assert out.json_int16_packed.tolist() == INT16_VECTOR_BYTES
    tm.assert_series_equal(out.text_int16_unpacked, INT16_SERIES, check_names=False)
    tm.assert_series_equal(out.json_int16_unpacked, INT16_SERIES, check_names=False)

    assert out.text_int32_packed.tolist() == INT32_VECTOR_BYTES
    assert out.json_int32_packed.tolist() == INT32_VECTOR_BYTES
    tm.assert_series_equal(out.text_int32_unpacked, INT32_SERIES, check_names=False)
    tm.assert_series_equal(out.json_int32_unpacked, INT32_SERIES, check_names=False)

    assert out.text_int64_packed.tolist() == INT64_VECTOR_BYTES
    assert out.json_int64_packed.tolist() == INT64_VECTOR_BYTES
    tm.assert_series_equal(out.text_int64_unpacked, INT64_SERIES, check_names=False)
    tm.assert_series_equal(out.json_int64_unpacked, INT64_SERIES, check_names=False)


def test_json_array_contains(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_contains_double=lambda x: x.text_vector.json_array_contains_double(0.1),
        text_contains_string=lambda x: x.text_list.json_array_contains_string('bar'),
        text_contains_json=lambda x: x.text_list.json_array_contains_json('true'),
        text_contains=lambda x: x.text_list.json_array_contains(True),

        json_contains_double=lambda x: x.json_vector.json_array_contains_double(0.1),
        json_contains_string=lambda x: x.json_list.json_array_contains_string('bar'),
        json_contains_json=lambda x: x.json_list.json_array_contains_json('true'),
        json_contains=lambda x: x.json_list.json_array_contains(True),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_contains_double.tolist() == [True, False, False]
    assert out.text_contains_string.tolist() == [False, True, False]
    assert out.text_contains_json.tolist() == [True, False, False]
    assert out.text_contains.tolist() == [True, False, False]

    assert out.json_contains_double.tolist() == [True, False, False]
    assert out.json_contains_string.tolist() == [False, True, False]
    assert out.json_contains_json.tolist() == [True, False, False]
    assert out.json_contains.tolist() == [True, False, False]


def test_json_array_push(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_vector_push_double=lambda x: x.text_vector.json_array_push_double(5.5),
        text_list_push_double=lambda x: x.text_list.json_array_push_double(5.5),
        text_obj_push_double=lambda x: x.text_obj.json_array_push_double(5.5),

        json_vector_push_double=lambda x: x.json_vector.json_array_push_double(5.5),
        json_list_push_double=lambda x: x.json_list.json_array_push_double(5.5),
        json_obj_push_double=lambda x: x.json_obj.json_array_push_double(5.5),

        text_vector_push_string=lambda x: x.text_vector.json_array_push_string('hi'),
        text_list_push_string=lambda x: x.text_list.json_array_push_string('hi'),
        text_obj_push_string=lambda x: x.text_obj.json_array_push_string('hi'),

        json_vector_push_string=lambda x: x.json_vector.json_array_push_string('hi'),
        json_list_push_string=lambda x: x.json_list.json_array_push_string('hi'),
        json_obj_push_string=lambda x: x.json_obj.json_array_push_string('hi'),

        text_vector_push_json=lambda x: x.text_vector.json_array_push_json('{"x": true}'),
        text_list_push_json=lambda x: x.text_list.json_array_push_json('{"x": true}'),
        text_obj_push_json=lambda x: x.text_obj.json_array_push_json('{"x": true}'),

        json_vector_push_json=lambda x: x.json_vector.json_array_push_json('{"x": true}'),
        json_list_push_json=lambda x: x.json_list.json_array_push_json('{"x": true}'),
        json_obj_push_json=lambda x: x.json_obj.json_array_push_json('{"x": true}'),

        json_vector_push=lambda x: x.json_vector.json_array_push({'x': True}),
        json_list_push=lambda x: x.json_list.json_array_push({'x': True}),
        json_obj_push=lambda x: x.json_obj.json_array_push({'x': True}),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_vector_push_double.tolist() == [x + [5.5] for x in FLOAT_VECTORS]
    assert out.text_list_push_double.tolist() == [x + [5.5] for x in LISTS]
    assert out.text_obj_push_double.tolist() == [None, None, None]

    assert out.json_vector_push_double.tolist() == [x + [5.5] for x in FLOAT_VECTORS]
    assert out.json_list_push_double.tolist() == [x + [5.5] for x in LISTS]
    assert out.json_obj_push_double.tolist() == [None, None, None]

    assert out.text_vector_push_string.tolist() == [x + ['hi'] for x in FLOAT_VECTORS]
    assert out.text_list_push_string.tolist() == [x + ['hi'] for x in LISTS]
    assert out.text_obj_push_string.tolist() == [None, None, None]

    assert out.json_vector_push_string.tolist() == [x + ['hi'] for x in FLOAT_VECTORS]
    assert out.json_list_push_string.tolist() == [x + ['hi'] for x in LISTS]
    assert out.json_obj_push_string.tolist() == [None, None, None]

    assert out.text_vector_push_json.tolist(
    ) == [x + [{'x': True}] for x in FLOAT_VECTORS]
    assert out.text_list_push_json.tolist() == [x + [{'x': True}] for x in LISTS]
    assert out.text_obj_push_json.tolist() == [None, None, None]

    assert out.json_vector_push_json.tolist(
    ) == [x + [{'x': True}] for x in FLOAT_VECTORS]
    assert out.json_list_push_json.tolist() == [x + [{'x': True}] for x in LISTS]
    assert out.json_obj_push_json.tolist() == [None, None, None]

    assert out.json_vector_push.tolist() == [x + [{'x': True}] for x in FLOAT_VECTORS]
    assert out.json_list_push.tolist() == [x + [{'x': True}] for x in LISTS]
    assert out.json_obj_push.tolist() == [None, None, None]


def test_json_delete_key(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_vector_delete_key=lambda x: x.text_vector.json_delete_key(1),
        json_vector_delete_key=lambda x: x.json_vector.json_delete_key(1),
        text_list_delete_key=lambda x: x.text_list.json_delete_key(1),
        json_list_delete_key=lambda x: x.json_list.json_delete_key(1),
        text_obj_delete_key=lambda x: x.text_obj.json_delete_key('b'),
        json_obj_delete_key=lambda x: x.json_obj.json_delete_key('b'),

        text_obj_delete_nested_key=lambda x: x.text_obj.json_delete_key('c', 'e'),
        json_obj_delete_nested_key=lambda x: x.json_obj.json_delete_key('c', 'e'),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    del_vector_key = [
        [0.1, -6.7],
        [10.3, 7.8],
        [-2.3, 7.9],
    ]
    del_list_key = [
        [0.1, True],
        [10.3, False],
        [0.3, False],
    ]
    del_obj_key = [
        {'a': 1, 'c': {'d': 100}},
        {'a': 1, 'c': {'d': 105, 'e': True}},
        {'a': 10, 'c': {'d': 111, 'e': 1.234}},
    ]
    del_nested_obj_key = [
        {'a': 1, 'b': 2, 'c': {'d': 100}},
        {'a': 1, 'b': 12, 'c': {'d': 105}},
        {'a': 10, 'b': 25, 'c': {'d': 111}},
    ]

    assert out.text_vector_delete_key.tolist() == del_vector_key
    assert out.json_vector_delete_key.tolist() == del_vector_key
    assert out.text_list_delete_key.tolist() == del_list_key
    assert out.json_list_delete_key.tolist() == del_list_key
    assert out.text_obj_delete_key.tolist() == del_obj_key
    assert out.json_obj_delete_key.tolist() == del_obj_key
    assert out.text_obj_delete_nested_key.tolist() == del_nested_obj_key
    assert out.json_obj_delete_nested_key.tolist() == del_nested_obj_key


def test_json_extract_double(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_vector_extract_double=lambda x: x.text_vector.json_extract_double(1),
        text_list_extract_double=lambda x: x.text_list.json_extract_double(0),
        text_obj_extract_double=lambda x: x.text_obj.json_extract_double('c', 'd'),

        json_vector_extract_double=lambda x: x.json_vector.json_extract_double(1),
        json_list_extract_double=lambda x: x.json_list.json_extract_double(0),
        json_obj_extract_double=lambda x: x.json_obj.json_extract_double('c', 'd'),

        text_vector_extract_string=lambda x: x.text_vector.json_extract_string(1),
        text_list_extract_string=lambda x: x.text_list.json_extract_string(1),
        text_obj_extract_string=lambda x: x.text_obj.json_extract_string('b'),

        json_vector_extract_string=lambda x: x.json_vector.json_extract_string(1),
        json_list_extract_string=lambda x: x.json_list.json_extract_string(1),
        json_obj_extract_string=lambda x: x.json_obj.json_extract_string('b'),

        text_vector_extract_json=lambda x: x.text_vector.json_extract_json(1),
        text_list_extract_json=lambda x: x.text_list.json_extract_json(0),
        text_obj_extract_json=lambda x: x.text_obj.json_extract_json('c'),

        json_vector_extract_json=lambda x: x.json_vector.json_extract_json(1),
        json_list_extract_json=lambda x: x.json_list.json_extract_json(0),
        json_obj_extract_json=lambda x: x.json_obj.json_extract_json('c'),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_vector_extract_double.tolist() == [2.5, -3.3, 6.6]
    assert out.text_list_extract_double.tolist() == [0.1, 10.3, 0.3]
    assert out.text_obj_extract_double.tolist() == [100, 105, 111]

    assert out.json_vector_extract_double.tolist() == [2.5, -3.3, 6.6]
    assert out.json_list_extract_double.tolist() == [0.1, 10.3, 0.3]
    assert out.json_obj_extract_double.tolist() == [100, 105, 111]

    assert out.text_vector_extract_string.tolist() == ['2.5', '-3.3', '6.6']
    assert out.text_list_extract_string.tolist() == ['foo', 'bar', 'baz']
    assert out.text_obj_extract_string.tolist() == ['2', '12', '25']

    assert out.json_vector_extract_string.tolist() == ['2.5', '-3.3', '6.6']
    assert out.json_list_extract_string.tolist() == ['foo', 'bar', 'baz']
    assert out.json_obj_extract_string.tolist() == ['2', '12', '25']

    objs = [
        {'d': 100},
        {'d': 105, 'e': True},
        {'d': 111, 'e': 1.234},
    ]

    assert out.text_vector_extract_json.tolist() == [2.5, -3.3, 6.6]
    assert out.text_list_extract_json.tolist() == [0.1, 10.3, 0.3]
    assert out.text_obj_extract_json.tolist() == objs

    assert out.json_vector_extract_json.tolist() == [2.5, -3.3, 6.6]
    assert out.json_list_extract_json.tolist() == [0.1, 10.3, 0.3]
    assert out.json_obj_extract_json.tolist() == objs


def test_json_get_type(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_get_type_double=lambda x: x.text_vector.json_extract_json(1).json_get_type(),
        json_get_type_double=lambda x: x.json_vector.json_extract_json(1).json_get_type(),

        text_get_type_string=lambda x: x.text_list.json_extract_json(1).json_get_type(),
        json_get_type_string=lambda x: x.json_list.json_extract_json(1).json_get_type(),

        text_get_type_object=lambda x: x.text_obj.json_extract_json('c').json_get_type(),
        json_get_type_object=lambda x: x.json_obj.json_extract_json('c').json_get_type(),

    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_get_type_double.tolist() == ['double', 'double', 'double']
    assert out.json_get_type_double.tolist() == ['double', 'double', 'double']

    assert out.text_get_type_string.tolist() == ['string', 'string', 'string']
    assert out.json_get_type_string.tolist() == ['string', 'string', 'string']

    assert out.text_get_type_object.tolist() == ['object', 'object', 'object']
    assert out.json_get_type_object.tolist() == ['object', 'object', 'object']


def test_json_mask(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_include_mask=lambda x: x.text_obj.json_include_mask('{"b":1}'),
        json_include_mask=lambda x: x.json_obj.json_include_mask('{"b":1}'),

        text_exclude_mask=lambda x: x.text_obj.json_exclude_mask('{"c":1}'),
        json_exclude_mask=lambda x: x.json_obj.json_exclude_mask('{"c":1}'),

        text_include_mask_py=lambda x: x.text_obj.json_include_mask({'b': 1}),
        json_include_mask_py=lambda x: x.json_obj.json_include_mask({'b': 1}),

        text_exclude_mask_py=lambda x: x.text_obj.json_exclude_mask({'c': 1}),
        json_exclude_mask_py=lambda x: x.json_obj.json_exclude_mask({'c': 1}),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_include_mask.tolist() == [{'b': 2}, {'b': 12}, {'b': 25}]
    assert out.json_include_mask.tolist() == [{'b': 2}, {'b': 12}, {'b': 25}]

    assert out.text_exclude_mask.tolist() == [
        {'a': 1, 'b': 2},
        {'a': 1, 'b': 12},
        {'a': 10, 'b': 25},
    ]
    assert out.json_exclude_mask.tolist() == [
        {'a': 1, 'b': 2},
        {'a': 1, 'b': 12},
        {'a': 10, 'b': 25},
    ]

    assert out.text_include_mask_py.tolist() == [{'b': 2}, {'b': 12}, {'b': 25}]
    assert out.json_include_mask_py.tolist() == [{'b': 2}, {'b': 12}, {'b': 25}]

    assert out.text_exclude_mask_py.tolist() == [
        {'a': 1, 'b': 2},
        {'a': 1, 'b': 12},
        {'a': 10, 'b': 25},
    ]
    assert out.json_exclude_mask_py.tolist() == [
        {'a': 1, 'b': 2},
        {'a': 1, 'b': 12},
        {'a': 10, 'b': 25},
    ]


def test_json_keys(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_keys=lambda x: x.text_obj.json_keys(),
        json_keys=lambda x: x.json_obj.json_keys(),

        text_nested_keys=lambda x: x.text_obj.json_keys('c'),
        json_nested_keys=lambda x: x.json_obj.json_keys('c'),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_keys.tolist() == [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]
    assert out.json_keys.tolist() == [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]

    assert out.text_nested_keys.tolist() == [['d'], ['d', 'e'], ['d', 'e']]
    assert out.json_nested_keys.tolist() == [['d'], ['d', 'e'], ['d', 'e']]


def test_json_length(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_length=lambda x: x.text_obj.json_length(),
        json_length=lambda x: x.json_obj.json_length(),

        text_nested_length=lambda x: x.text_obj.json_extract_json('c').json_length(),
        json_nested_length=lambda x: x.json_obj.json_extract_json('c').json_length(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_length.tolist() == [3, 3, 3]
    assert out.json_length.tolist() == [3, 3, 3]

    assert out.text_nested_length.tolist() == [1, 2, 2]
    assert out.json_nested_length.tolist() == [1, 2, 2]


def test_json_pretty(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_pretty=lambda x: x.text_obj.json_pretty(),
        json_pretty=lambda x: x.json_obj.json_pretty(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    pretty = [
        '''{
  "a": 1,
  "b": 2,
  "c": {
    "d": 100
  }
}''',
        '''{
  "a": 1,
  "b": 12,
  "c": {
    "d": 105,
    "e": true
  }
}''',
        '''{
  "a": 10,
  "b": 25,
  "c": {
    "d": 111,
    "e": 1.234
  }
}''',
    ]

    assert out.text_pretty.tolist() == pretty
    assert out.json_pretty.tolist() == pretty


def test_json_set_double(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_set_double=lambda x: x.text_obj.json_set_double('f', 3.14),
        json_set_double=lambda x: x.json_obj.json_set_double('f', 3.14),

        text_set_array_double=lambda x: x.text_list.json_set_double(1, 3.14),
        json_set_array_double=lambda x: x.json_list.json_set_double(1, 3.14),

        text_set_nested_double=lambda x: x.text_obj.json_set_double('c', 'f', 3.14),
        json_set_nested_double=lambda x: x.json_obj.json_set_double('c', 'f', 3.14),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    values = out.text_set_double.tolist()
    assert values[0]['f'] == 3.14
    assert values[1]['f'] == 3.14
    assert values[2]['f'] == 3.14

    values = out.json_set_double.tolist()
    assert values[0]['f'] == 3.14
    assert values[1]['f'] == 3.14
    assert values[2]['f'] == 3.14

    values = out.text_set_nested_double.tolist()
    assert values[0]['c']['f'] == 3.14
    assert values[1]['c']['f'] == 3.14
    assert values[2]['c']['f'] == 3.14

    values = out.json_set_nested_double.tolist()
    assert values[0]['c']['f'] == 3.14
    assert values[1]['c']['f'] == 3.14
    assert values[2]['c']['f'] == 3.14

    assert out.text_set_array_double.tolist() == [
        [0.1, 3.14, True],
        [10.3, 3.14, False],
        [0.3, 3.14, False],
    ]
    assert out.json_set_array_double.tolist() == [
        [0.1, 3.14, True],
        [10.3, 3.14, False],
        [0.3, 3.14, False],
    ]


def test_json_set_string(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_set_string=lambda x: x.text_obj.json_set_string('f', '3.14'),
        json_set_string=lambda x: x.json_obj.json_set_string('f', '3.14'),

        text_set_array_string=lambda x: x.text_list.json_set_string(1, '3.14'),
        json_set_array_string=lambda x: x.json_list.json_set_string(1, '3.14'),

        text_set_nested_string=lambda x: x.text_obj.json_set_string('c', 'f', '3.14'),
        json_set_nested_string=lambda x: x.json_obj.json_set_string('c', 'f', '3.14'),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    values = out.text_set_string.tolist()
    assert values[0]['f'] == '3.14'
    assert values[1]['f'] == '3.14'
    assert values[2]['f'] == '3.14'

    values = out.json_set_string.tolist()
    assert values[0]['f'] == '3.14'
    assert values[1]['f'] == '3.14'
    assert values[2]['f'] == '3.14'

    values = out.text_set_nested_string.tolist()
    assert values[0]['c']['f'] == '3.14'
    assert values[1]['c']['f'] == '3.14'
    assert values[2]['c']['f'] == '3.14'

    values = out.json_set_nested_string.tolist()
    assert values[0]['c']['f'] == '3.14'
    assert values[1]['c']['f'] == '3.14'
    assert values[2]['c']['f'] == '3.14'

    assert out.text_set_array_string.tolist() == [
        [0.1, '3.14', True],
        [10.3, '3.14', False],
        [0.3, '3.14', False],
    ]
    assert out.json_set_array_string.tolist() == [
        [0.1, '3.14', True],
        [10.3, '3.14', False],
        [0.3, '3.14', False],
    ]


def test_json_set_json(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_set_json=lambda x: x.text_obj.json_set_json('f', 'true'),
        json_set_json=lambda x: x.json_obj.json_set_json('f', 'true'),

        text_set_json_obj=lambda x: x.text_obj.json_set_json('f', '{"x":1000}'),
        json_set_json_obj=lambda x: x.json_obj.json_set_json('f', '{"x":1000}'),

        text_set_array_json=lambda x: x.text_list.json_set_json(1, 'true'),
        json_set_array_json=lambda x: x.json_list.json_set_json(1, 'true'),

        text_set_nested_json=lambda x: x.text_obj.json_set_json('c', 'f', 'true'),
        json_set_nested_json=lambda x: x.json_obj.json_set_json('c', 'f', 'true'),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    values = out.text_set_json.tolist()
    assert values[0]['f'] is True
    assert values[1]['f'] is True
    assert values[2]['f'] is True

    values = out.json_set_json.tolist()
    assert values[0]['f'] is True
    assert values[1]['f'] is True
    assert values[2]['f'] is True

    values = out.text_set_json_obj.tolist()
    assert values[0]['f'] == dict(x=1000)
    assert values[1]['f'] == dict(x=1000)
    assert values[2]['f'] == dict(x=1000)

    values = out.json_set_json_obj.tolist()
    assert values[0]['f'] == dict(x=1000)
    assert values[1]['f'] == dict(x=1000)
    assert values[2]['f'] == dict(x=1000)

    values = out.text_set_nested_json.tolist()
    assert values[0]['c']['f'] is True
    assert values[1]['c']['f'] is True
    assert values[2]['c']['f'] is True

    values = out.json_set_nested_json.tolist()
    assert values[0]['c']['f'] is True
    assert values[1]['c']['f'] is True
    assert values[2]['c']['f'] is True

    assert out.text_set_array_json.tolist() == [
        [0.1, True, True],
        [10.3, True, False],
        [0.3, True, False],
    ]
    assert out.json_set_array_json.tolist() == [
        [0.1, True, True],
        [10.3, True, False],
        [0.3, True, False],
    ]


def test_json_set(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_set=lambda x: x.text_obj.json_set('f', True),
        json_set=lambda x: x.json_obj.json_set('f', True),

        text_set_obj=lambda x: x.text_obj.json_set('f', {'x': 1000}),
        json_set_obj=lambda x: x.json_obj.json_set('f', {'x': 1000}),

        text_set_array=lambda x: x.text_list.json_set(1, True),
        json_set_array=lambda x: x.json_list.json_set(1, True),

        text_set_nested=lambda x: x.text_obj.json_set('c', 'f', True),
        json_set_nested=lambda x: x.json_obj.json_set('c', 'f', True),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    values = out.text_set.tolist()
    assert values[0]['f'] is True
    assert values[1]['f'] is True
    assert values[2]['f'] is True

    values = out.json_set.tolist()
    assert values[0]['f'] is True
    assert values[1]['f'] is True
    assert values[2]['f'] is True

    values = out.text_set_obj.tolist()
    assert values[0]['f'] == dict(x=1000)
    assert values[1]['f'] == dict(x=1000)
    assert values[2]['f'] == dict(x=1000)

    values = out.json_set_obj.tolist()
    assert values[0]['f'] == dict(x=1000)
    assert values[1]['f'] == dict(x=1000)
    assert values[2]['f'] == dict(x=1000)

    values = out.text_set_nested.tolist()
    assert values[0]['c']['f'] is True
    assert values[1]['c']['f'] is True
    assert values[2]['c']['f'] is True

    values = out.json_set_nested.tolist()
    assert values[0]['c']['f'] is True
    assert values[1]['c']['f'] is True
    assert values[2]['c']['f'] is True

    assert out.text_set_array.tolist() == [
        [0.1, True, True],
        [10.3, True, False],
        [0.3, True, False],
    ]
    assert out.json_set_array.tolist() == [
        [0.1, True, True],
        [10.3, True, False],
        [0.3, True, False],
    ]


def test_json_splice(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        text_splice_double=lambda x: x.text_list.json_splice_double(1, 3, 3.14),
        json_splice_double=lambda x: x.json_list.json_splice_double(1, 3, 3.14),

        text_splice_string=lambda x: x.text_list.json_splice_string(1, 3, '3.14'),
        json_splice_string=lambda x: x.json_list.json_splice_string(1, 3, '3.14'),

        text_splice_json=lambda x: x.text_list.json_splice_json(1, 3, '{"x": 3.14}'),
        json_splice_json=lambda x: x.json_list.json_splice_json(1, 3, '{"x": 3.14}'),

        text_splice=lambda x: x.text_list.json_splice(1, 3, {'x': 3.14}),
        json_splice=lambda x: x.json_list.json_splice(1, 3, {'x': 3.14}),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.text_splice_double.tolist() == [[0.1, 3.14], [10.3, 3.14], [0.3, 3.14]]
    assert out.json_splice_double.tolist() == [[0.1, 3.14], [10.3, 3.14], [0.3, 3.14]]

    assert out.text_splice_string.tolist() == [
        [0.1, '3.14'], [
            10.3, '3.14',
        ], [0.3, '3.14'],
    ]
    assert out.json_splice_string.tolist() == [
        [0.1, '3.14'], [
            10.3, '3.14',
        ], [0.3, '3.14'],
    ]

    assert out.text_splice_json.tolist() == [
        [0.1, {'x': 3.14}],
        [10.3, {'x': 3.14}],
        [0.3, {'x': 3.14}],
    ]
    assert out.json_splice_json.tolist() == [
        [0.1, {'x': 3.14}],
        [10.3, {'x': 3.14}],
        [0.3, {'x': 3.14}],
    ]

    assert out.text_splice.tolist() == [
        [0.1, {'x': 3.14}],
        [10.3, {'x': 3.14}],
        [0.3, {'x': 3.14}],
    ]
    assert out.json_splice.tolist() == [
        [0.1, {'x': 3.14}],
        [10.3, {'x': 3.14}],
        [0.3, {'x': 3.14}],
    ]


def test_bitcount(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        int_bitcount=lambda x: x.int_c.bit_count(),
        bigint_bitcount=lambda x: x.bigint_c.bit_count(),
        float_bitcount=lambda x: x.float_c.bit_count(),
        double_bitcount=lambda x: x.double_c.bit_count(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    out['py_int_bitcount'] = out['int_c'].apply(lambda x: format(x, 'b').count('1'))
    out['py_bigint_bitcount'] = out['bigint_c'].apply(lambda x: format(x, 'b').count('1'))
    out['py_float_bitcount'] = out['float_c'].apply(
        lambda x: format(round(x), 'b').count('1'),
    )
    out['py_double_bitcount'] = out['double_c'].apply(
        lambda x: format(round(x), 'b').count('1'),
    )

    assert out.int_bitcount.tolist() == out.py_int_bitcount.tolist()
    assert out.bigint_bitcount.tolist() == out.py_bigint_bitcount.tolist()
    assert out.float_bitcount.tolist() == out.py_float_bitcount.tolist()
    assert out.double_bitcount.tolist() == out.py_double_bitcount.tolist()


def test_conv(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        int_hex=lambda x: x.int_c.conv(10, 16),
        bigint_hex=lambda x: x.bigint_c.conv(10, 16),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    out['py_int_hex'] = out['int_c'].apply(lambda x: np.base_repr(x, base=16))
    out['py_bigint_hex'] = out['bigint_c'].apply(lambda x: np.base_repr(x, base=16))

    assert out.int_hex.tolist() == out.py_int_hex.tolist()
    assert out.bigint_hex.tolist() == out.py_bigint_hex.tolist()


def test_sigmoid(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        int_sigmoid=lambda x: x.int_c.sigmoid(),
        bigint_sigmoid=lambda x: x.bigint_c.sigmoid(),
        float_sigmoid=lambda x: x.float_c.sigmoid(),
        double_sigmoid=lambda x: x.double_c.sigmoid(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    def sigmoid(x: Any) -> Any:
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

    out['py_int_sigmoid'] = out['int_c'].apply(sigmoid)
    out['py_bigint_sigmoid'] = out['bigint_c'].apply(sigmoid)
    out['py_float_sigmoid'] = out['float_c'].apply(sigmoid)
    out['py_double_sigmoid'] = out['double_c'].apply(sigmoid)

    assert out.int_sigmoid.tolist() == out.py_int_sigmoid.tolist()
    assert out.bigint_sigmoid.tolist() == out.py_bigint_sigmoid.tolist()
    assert out.float_sigmoid.tolist() == out.py_float_sigmoid.tolist()
    assert out.double_sigmoid.tolist() == out.py_double_sigmoid.tolist()


def test_to_number(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        parse_int=lambda x: x.int_c.cast('string').to_number(),
        parse_bigint=lambda x: x.bigint_c.cast('string').to_number(),
        parse_float=lambda x: x.float_c.cast('string').to_number(),
        parse_double=lambda x: x.double_c.cast('string').to_number(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    tm.assert_series_equal(
        out.parse_int, pd.Series(
            [41.0, 44.0, 30.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.parse_bigint, pd.Series(
            [12.0, 46.0, 65.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.parse_float, pd.Series(
            [88.8983, 11.4726, 13.5833],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.parse_double, pd.Series(
            [49.002318, 87.812455, 58.052145],
        ), check_names=False,
    )


def test_trunc(con: Any) -> None:
    tbl = con.table('datatypes')

    out = tbl.mutate(
        trunc_float=lambda x: x.float_c.trunc(),
        trunc_double=lambda x: x.double_c.trunc(),

        trunc_float_2=lambda x: x.float_c.trunc(2),
        trunc_double_2=lambda x: x.double_c.trunc(2),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    tm.assert_series_equal(
        out.trunc_float, pd.Series(
            [88.0, 11.0, 13.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.trunc_double, pd.Series(
            [49.0, 87.0, 58.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.trunc_float_2, pd.Series(
            [88.89, 11.47, 13.58],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.trunc_double_2, pd.Series(
            [49.00, 87.81, 58.05],
        ), check_names=False,
    )


def test_truncate(con: Any) -> None:
    tbl = con.table('datatypes')

    with pytest.raises(TypeError):
        tbl.float_c.truncate()

    out = tbl.mutate(
        truncate_float_2=lambda x: x.float_c.truncate(2),
        truncate_double_2=lambda x: x.double_c.truncate(2),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    tm.assert_series_equal(
        out.truncate_float_2, pd.Series(
            [88.89, 11.47, 13.58],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.truncate_double_2, pd.Series(
            [49.00, 87.81, 58.05],
        ), check_names=False,
    )


def test_dot_product(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        dp=lambda x: x.vec1.dot_product(x.vec2),
        dp_f32=lambda x: x.vec1_f32.dot_product_f32(x.vec2_f32),
        dp_f64=lambda x: x.vec1_f64.dot_product_f64(x.vec2_f64),
        dp_i8=lambda x: x.vec1_i8.dot_product_i8(x.vec2_i8),
        dp_i16=lambda x: x.vec1_i16.dot_product_i16(x.vec2_i16),
        dp_i32=lambda x: x.vec1_i32.dot_product_i32(x.vec2_i32),
        dp_i64=lambda x: x.vec1_i64.dot_product_i64(x.vec2_i64),

        dp_lit=lambda x: x.vec1.dot_product([1, 2, 3]),
        dp_f32_lit=lambda x: x.vec1_f32.dot_product_f32([1, 2, 3]),
        dp_f64_lit=lambda x: x.vec1_f64.dot_product_f64([1, 2, 3]),
        dp_i8_lit=lambda x: x.vec1_i8.dot_product_i8([1, 2, 3]),
        dp_i16_lit=lambda x: x.vec1_i16.dot_product_i16([1, 2, 3]),
        dp_i32_lit=lambda x: x.vec1_i32.dot_product_i32([1, 2, 3]),
        dp_i64_lit=lambda x: x.vec1_i64.dot_product_i64([1, 2, 3]),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    tm.assert_series_equal(
        out.dp, pd.Series(
            [51.149994, 177.820007, 111.259995],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_f32, pd.Series(
            [51.149994, 177.820007, 111.259995],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_f64, pd.Series(
            [51.15, 177.82, 111.26],
        ), check_names=False,
    )
    assert out.dp_i8.tolist() == [58.0, 173.0, 117.0]
    assert out.dp_i16.tolist() == [58.0, 173.0, 117.0]
    assert out.dp_i32.tolist() == [58.0, 173.0, 117.0]
    assert out.dp_i64.tolist() == [58.0, 173.0, 117.0]

    tm.assert_series_equal(
        out.dp_lit, pd.Series(
            [-14.999998, 27.100002, 34.599998],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_f32_lit, pd.Series(
            [-14.999998, 27.100002, 34.599998],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_f64_lit, pd.Series(
            [-15.0, 27.1, 34.6],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_i8_lit, pd.Series(
            [-15.0, 28.0, 36.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_i16_lit, pd.Series(
            [-15.0, 28.0, 36.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_i32_lit, pd.Series(
            [-15.0, 28.0, 36.0],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.dp_i64_lit, pd.Series(
            [-15.0, 28.0, 36.0],
        ), check_names=False,
    )


def test_euclidean_distance(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        ed=lambda x: x.vec1.euclidean_distance(x.vec2),
        ed_f32=lambda x: x.vec1_f32.euclidean_distance_f32(x.vec2_f32),
        ed_f64=lambda x: x.vec1_f64.euclidean_distance_f64(x.vec2_f64),
        ed_i8=lambda x: x.vec1_i8.euclidean_distance_i8(x.vec2_i8),
        ed_i16=lambda x: x.vec1_i16.euclidean_distance_i16(x.vec2_i16),
        ed_i32=lambda x: x.vec1_i32.euclidean_distance_i32(x.vec2_i32),
        ed_i64=lambda x: x.vec1_i64.euclidean_distance_i64(x.vec2_i64),

        ed_lit=lambda x: x.vec1.euclidean_distance([1, 2, 3]),
        ed_f32_lit=lambda x: x.vec1_f32.euclidean_distance_f32([1, 2, 3]),
        ed_f64_lit=lambda x: x.vec1_f64.euclidean_distance_f64([1, 2, 3]),
        ed_i8_lit=lambda x: x.vec1_i8.euclidean_distance_i8([1, 2, 3]),
        ed_i16_lit=lambda x: x.vec1_i16.euclidean_distance_i16([1, 2, 3]),
        ed_i32_lit=lambda x: x.vec1_i32.euclidean_distance_i32([1, 2, 3]),
        ed_i64_lit=lambda x: x.vec1_i64.euclidean_distance_i64([1, 2, 3]),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    tm.assert_series_equal(out.ed, pd.Series([0.0, 0.0, 0.0]), check_names=False)
    tm.assert_series_equal(out.ed_f32, pd.Series([0.0, 0.0, 0.0]), check_names=False)
    tm.assert_series_equal(out.ed_f64, pd.Series([0.0, 0.0, 0.0]), check_names=False)
    assert out.ed_i8.tolist() == [0.0, 0.0, 0.0]
    assert out.ed_i16.tolist() == [0.0, 0.0, 0.0]
    assert out.ed_i32.tolist() == [0.0, 0.0, 0.0]
    assert out.ed_i64.tolist() == [0.0, 0.0, 0.0]

    tm.assert_series_equal(
        out.ed_lit, pd.Series(
            [9.754486, 11.731156, 7.487322],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_f32_lit, pd.Series(
            [9.754486, 11.731156, 7.487322],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_f64_lit, pd.Series(
            [9.754486, 11.731156, 7.487322],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_i8_lit, pd.Series(
            [10.099505, 11.445523, 7.681146],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_i16_lit, pd.Series(
            [10.099505, 11.445523, 7.681146],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_i32_lit, pd.Series(
            [10.099505, 11.445523, 7.681146],
        ), check_names=False,
    )
    tm.assert_series_equal(
        out.ed_i64_lit, pd.Series(
            [10.099505, 11.445523, 7.681146],
        ), check_names=False,
    )


def test_scalar_vector_mul(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        ed=lambda x: x.vec1.scalar_vector_mul(3).json_array_unpack(),
        ed_f32=lambda x: x.vec1_f32.scalar_vector_mul_f32(3).json_array_unpack_f32(),
        ed_f64=lambda x: x.vec1_f64.scalar_vector_mul_f64(3).json_array_unpack_f64(),
        ed_i8=lambda x: x.vec1_i8.scalar_vector_mul_i8(3).json_array_unpack_i8(),
        ed_i16=lambda x: x.vec1_i16.scalar_vector_mul_i16(3).json_array_unpack_i16(),
        ed_i32=lambda x: x.vec1_i32.scalar_vector_mul_i32(3).json_array_unpack_i32(),
        ed_i64=lambda x: x.vec1_i64.scalar_vector_mul_i64(3).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([[y * 3 for y in x] for x in FLOAT_VECTORS])
    ints = pd.Series([[y * 3 for y in x] for x in INT_VECTORS])

    tm.assert_series_equal(out.ed, floats, check_names=False)
    tm.assert_series_equal(out.ed_f32, floats, check_names=False)
    tm.assert_series_equal(out.ed_f64, floats, check_names=False)
    tm.assert_series_equal(out.ed_i8, ints, check_names=False)
    tm.assert_series_equal(out.ed_i16, ints, check_names=False)
    tm.assert_series_equal(out.ed_i32, ints, check_names=False)
    tm.assert_series_equal(out.ed_i64, ints, check_names=False)


def test_vector_add(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        va=lambda x: x.vec1.vector_add(x.vec2).json_array_unpack(),
        va_f32=lambda x: x.vec1_f32.vector_add_f32(x.vec2_f32).json_array_unpack_f32(),
        va_f64=lambda x: x.vec1_f64.vector_add_f64(x.vec2_f64).json_array_unpack_f64(),
        va_i8=lambda x: x.vec1_i8.vector_add_i8(x.vec2_i8).json_array_unpack_i8(),
        va_i16=lambda x: x.vec1_i16.vector_add_i16(x.vec2_i16).json_array_unpack_i16(),
        va_i32=lambda x: x.vec1_i32.vector_add_i32(x.vec2_i32).json_array_unpack_i32(),
        va_i64=lambda x: x.vec1_i64.vector_add_i64(x.vec2_i64).json_array_unpack_i64(),

        va_lit=lambda x: x.vec1.vector_add([1.0, 2.0, 3.0]).json_array_unpack(),
        va_f32_lit=lambda x: x.vec1_f32.vector_add_f32(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_f32(),
        va_f64_lit=lambda x: x.vec1_f64.vector_add_f64(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_f64(),
        va_i8_lit=lambda x: x.vec1_i8.vector_add_i8(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i8(),
        va_i16_lit=lambda x: x.vec1_i16.vector_add_i16(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i16(),
        va_i32_lit=lambda x: x.vec1_i32.vector_add_i32(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i32(),
        va_i64_lit=lambda x: x.vec1_i64.vector_add_i64(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([
        [0.200000003, 5, -13.3999996],
        [20.6000004, -6.5999999, 15.6000004],
        [-4.5999999, 13.1999998, 15.8000002],
    ])
    ints = pd.Series([[0, 6, -14], [20, -6, 16], [-4, 14, 16]])

    tm.assert_series_equal(out.va, floats, check_names=False)
    tm.assert_series_equal(out.va_f32, floats, check_names=False)
    tm.assert_series_equal(out.va_f64, floats, check_names=False)
    tm.assert_series_equal(out.va_i8, ints, check_names=False)
    tm.assert_series_equal(out.va_i16, ints, check_names=False)
    tm.assert_series_equal(out.va_i32, ints, check_names=False)
    tm.assert_series_equal(out.va_i64, ints, check_names=False)

    floats = pd.Series([[x[0] + 1, x[1] + 2, x[2] + 3] for x in FLOAT_VECTORS])
    ints = pd.Series([[x[0] + 1, x[1] + 2, x[2] + 3] for x in INT_VECTORS])

    tm.assert_series_equal(out.va_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_f32_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_f64_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_i8_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i16_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i32_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i64_lit, ints, check_names=False)


def test_vector_elements_sum(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        vs=lambda x: x.vec1.vector_elements_sum(),
        vs_f32=lambda x: x.vec1_f32.vector_elements_sum_f32(),
        vs_f64=lambda x: x.vec1_f64.vector_elements_sum_f64(),
        vs_i8=lambda x: x.vec1_i8.vector_elements_sum_i8(),
        vs_i16=lambda x: x.vec1_i16.vector_elements_sum_i16(),
        vs_i32=lambda x: x.vec1_i32.vector_elements_sum_i32(),
        vs_i64=lambda x: x.vec1_i64.vector_elements_sum_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([-4.1, 14.8, 12.2])
    ints = pd.Series([-4.0, 15.0, 13.0])

    tm.assert_series_equal(out.vs, floats, check_names=False)
    tm.assert_series_equal(out.vs_f32, floats, check_names=False)
    tm.assert_series_equal(out.vs_f64, floats, check_names=False)
    tm.assert_series_equal(out.vs_i8, ints, check_names=False)
    tm.assert_series_equal(out.vs_i16, ints, check_names=False)
    tm.assert_series_equal(out.vs_i32, ints, check_names=False)
    tm.assert_series_equal(out.vs_i64, ints, check_names=False)


def test_vector_kth_element(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        ve=lambda x: x.vec1.vector_kth_element(1),
        ve_f32=lambda x: x.vec1_f32.vector_kth_element_f32(1),
        ve_f64=lambda x: x.vec1_f64.vector_kth_element_f64(1),
        ve_i8=lambda x: x.vec1_i8.vector_kth_element_i8(1),
        ve_i16=lambda x: x.vec1_i16.vector_kth_element_i16(1),
        ve_i32=lambda x: x.vec1_i32.vector_kth_element_i32(1),
        ve_i64=lambda x: x.vec1_i64.vector_kth_element_i64(1),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([2.5, -3.3, 6.6])
    ints = pd.Series([3.0, -3.0, 7.0])

    tm.assert_series_equal(out.ve, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_f32, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_f64, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_i8, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_i16, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_i32, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.ve_i64, ints, check_names=False, check_dtype=False)


def test_vector_mul(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        vm=lambda x: x.vec1.vector_mul(x.vec2).json_array_unpack(),
        vm_f32=lambda x: x.vec1_f32.vector_mul_f32(x.vec2_f32).json_array_unpack_f32(),
        vm_f64=lambda x: x.vec1_f64.vector_mul_f64(x.vec2_f64).json_array_unpack_f64(),
        vm_i8=lambda x: x.vec1_i8.vector_mul_i8(x.vec2_i8).json_array_unpack_i8(),
        vm_i16=lambda x: x.vec1_i16.vector_mul_i16(x.vec2_i16).json_array_unpack_i16(),
        vm_i32=lambda x: x.vec1_i32.vector_mul_i32(x.vec2_i32).json_array_unpack_i32(),
        vm_i64=lambda x: x.vec1_i64.vector_mul_i64(x.vec2_i64).json_array_unpack_i64(),

        vm_lit=lambda x: x.vec1.vector_mul([1, 2, 3]).json_array_unpack(),
        vm_f32_lit=lambda x: x.vec1_f32.vector_mul_f32([1, 2, 3]).json_array_unpack_f32(),
        vm_f64_lit=lambda x: x.vec1_f64.vector_mul_f64([1, 2, 3]).json_array_unpack_f64(),
        vm_i8_lit=lambda x: x.vec1_i8.vector_mul_i8([1, 2, 3]).json_array_unpack_i8(),
        vm_i16_lit=lambda x: x.vec1_i16.vector_mul_i16([1, 2, 3]).json_array_unpack_i16(),
        vm_i32_lit=lambda x: x.vec1_i32.vector_mul_i32([1, 2, 3]).json_array_unpack_i32(),
        vm_i64_lit=lambda x: x.vec1_i64.vector_mul_i64([1, 2, 3]).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([
        [0.0100000007, 6.25, 44.8899956],
        [106.090004, 10.8899994, 60.840004],
        [5.28999996, 43.5599976, 62.4099998],
    ])
    ints = pd.Series([[0, 9, 49], [100, 9, 64], [4, 49, 64]])

    # TODO: Bug in vector_mul?!?
#   tm.assert_series_equal(out.vm, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_f32, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_f64, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i8, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i16, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i32, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i64, ints, check_names=False, check_dtype=False)

    floats = pd.Series([
        [0.100000001, 5, -20.0999985],
        [10.3000002, -6.5999999, 23.4000015],
        [-2.29999995, 13.1999998, 23.7000008],
    ])
    ints = pd.Series([[0, 6, -21], [10, -6, 24], [-2, 14, 24]])

    # TODO: Bug in vector_mul?!?
#   tm.assert_series_equal(out.vm_lit, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_f32_lit, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_f64_lit, floats, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i8_lit, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i16_lit, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i32_lit, ints, check_names=False, check_dtype=False)
    tm.assert_series_equal(out.vm_i64_lit, ints, check_names=False, check_dtype=False)


def test_vector_num_elements(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        vm=lambda x: x.vec1.vector_num_elements(),
        vm_f32=lambda x: x.vec1_f32.vector_num_elements_f32(),
        vm_f64=lambda x: x.vec1_f64.vector_num_elements_f64(),
        vm_i8=lambda x: x.vec1_i8.vector_num_elements_i8(),
        vm_i16=lambda x: x.vec1_i16.vector_num_elements_i16(),
        vm_i32=lambda x: x.vec1_i32.vector_num_elements_i32(),
        vm_i64=lambda x: x.vec1_i64.vector_num_elements_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    assert out.vm.tolist() == [3, 3, 3]
    assert out.vm_f32.tolist() == [3, 3, 3]
    assert out.vm_f64.tolist() == [3, 3, 3]
    assert out.vm_i8.tolist() == [3, 3, 3]
    assert out.vm_i16.tolist() == [3, 3, 3]
    assert out.vm_i32.tolist() == [3, 3, 3]
    assert out.vm_i64.tolist() == [3, 3, 3]


def test_vector_sort(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        vs=lambda x: x.vec1.vector_sort().json_array_unpack(),
        vs_f32=lambda x: x.vec1_f32.vector_sort_f32().json_array_unpack_f32(),
        vs_f64=lambda x: x.vec1_f64.vector_sort_f64().json_array_unpack_f64(),
        vs_i8=lambda x: x.vec1_i8.vector_sort_i8().json_array_unpack_i8(),
        vs_i16=lambda x: x.vec1_i16.vector_sort_i16().json_array_unpack_i16(),
        vs_i32=lambda x: x.vec1_i32.vector_sort_i32().json_array_unpack_i32(),
        vs_i64=lambda x: x.vec1_i64.vector_sort_i64().json_array_unpack_i64(),

        vs_dir=lambda x: x.vec1.vector_sort('desc').json_array_unpack(),
        vs_f32_dir=lambda x: x.vec1_f32.vector_sort_f32('desc').json_array_unpack_f32(),
        vs_f64_dir=lambda x: x.vec1_f64.vector_sort_f64('desc').json_array_unpack_f64(),
        vs_i8_dir=lambda x: x.vec1_i8.vector_sort_i8('desc').json_array_unpack_i8(),
        vs_i16_dir=lambda x: x.vec1_i16.vector_sort_i16('desc').json_array_unpack_i16(),
        vs_i32_dir=lambda x: x.vec1_i32.vector_sort_i32('desc').json_array_unpack_i32(),
        vs_i64_dir=lambda x: x.vec1_i64.vector_sort_i64('desc').json_array_unpack_i64(),

        vs_idir=lambda x: x.vec1.vector_sort(ibis.desc).json_array_unpack(),
        vs_f32_idir=lambda x: x.vec1_f32.vector_sort_f32(
            ibis.desc,
        ).json_array_unpack_f32(),
        vs_f64_idir=lambda x: x.vec1_f64.vector_sort_f64(
            ibis.desc,
        ).json_array_unpack_f64(),
        vs_i8_idir=lambda x: x.vec1_i8.vector_sort_i8(ibis.desc).json_array_unpack_i8(),
        vs_i16_idir=lambda x: x.vec1_i16.vector_sort_i16(
            ibis.desc,
        ).json_array_unpack_i16(),
        vs_i32_idir=lambda x: x.vec1_i32.vector_sort_i32(
            ibis.desc,
        ).json_array_unpack_i32(),
        vs_i64_idir=lambda x: x.vec1_i64.vector_sort_i64(
            ibis.desc,
        ).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([
        [-6.69999981, 0.100000001, 2.5],
        [-3.29999995, 7.80000019, 10.3000002],
        [-2.29999995, 6.5999999, 7.9000001],
    ])
    ints = pd.Series([[-7, 0, 3], [-3, 8, 10], [-2, 7, 8]])

    tm.assert_series_equal(out.vs, floats, check_names=False)
    tm.assert_series_equal(out.vs_f32, floats, check_names=False)
    tm.assert_series_equal(out.vs_f64, floats, check_names=False)
    tm.assert_series_equal(out.vs_i8, ints, check_names=False)
    tm.assert_series_equal(out.vs_i16, ints, check_names=False)
    tm.assert_series_equal(out.vs_i32, ints, check_names=False)
    tm.assert_series_equal(out.vs_i64, ints, check_names=False)

    floats = pd.Series([
        list(reversed([-6.69999981, 0.100000001, 2.5])),
        list(reversed([-3.29999995, 7.80000019, 10.3000002])),
        list(reversed([-2.29999995, 6.5999999, 7.9000001])),
    ])
    ints = pd.Series([
        list(reversed([-7, 0, 3])),
        list(reversed([-3, 8, 10])),
        list(reversed([-2, 7, 8])),
    ])

    tm.assert_series_equal(out.vs_dir, floats, check_names=False)
    tm.assert_series_equal(out.vs_f32_dir, floats, check_names=False)
    tm.assert_series_equal(out.vs_f64_dir, floats, check_names=False)
    tm.assert_series_equal(out.vs_i8_dir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i16_dir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i32_dir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i64_dir, ints, check_names=False)

    tm.assert_series_equal(out.vs_idir, floats, check_names=False)
    tm.assert_series_equal(out.vs_f32_idir, floats, check_names=False)
    tm.assert_series_equal(out.vs_f64_idir, floats, check_names=False)
    tm.assert_series_equal(out.vs_i8_idir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i16_idir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i32_idir, ints, check_names=False)
    tm.assert_series_equal(out.vs_i64_dir, ints, check_names=False)


def test_vector_sub(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        va=lambda x: x.vec1.vector_sub(x.vec2).json_array_unpack(),
        va_f32=lambda x: x.vec1_f32.vector_sub_f32(x.vec2_f32).json_array_unpack_f32(),
        va_f64=lambda x: x.vec1_f64.vector_sub_f64(x.vec2_f64).json_array_unpack_f64(),
        va_i8=lambda x: x.vec1_i8.vector_sub_i8(x.vec2_i8).json_array_unpack_i8(),
        va_i16=lambda x: x.vec1_i16.vector_sub_i16(x.vec2_i16).json_array_unpack_i16(),
        va_i32=lambda x: x.vec1_i32.vector_sub_i32(x.vec2_i32).json_array_unpack_i32(),
        va_i64=lambda x: x.vec1_i64.vector_sub_i64(x.vec2_i64).json_array_unpack_i64(),

        va_lit=lambda x: x.vec1.vector_sub([1.0, 2.0, 3.0]).json_array_unpack(),
        va_f32_lit=lambda x: x.vec1_f32.vector_sub_f32(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_f32(),
        va_f64_lit=lambda x: x.vec1_f64.vector_sub_f64(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_f64(),
        va_i8_lit=lambda x: x.vec1_i8.vector_sub_i8(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i8(),
        va_i16_lit=lambda x: x.vec1_i16.vector_sub_i16(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i16(),
        va_i32_lit=lambda x: x.vec1_i32.vector_sub_i32(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i32(),
        va_i64_lit=lambda x: x.vec1_i64.vector_sub_i64(
            [1.0, 2.0, 3.0],
        ).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    ints = pd.Series([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    tm.assert_series_equal(out.va, floats, check_names=False)
    tm.assert_series_equal(out.va_f32, floats, check_names=False)
    tm.assert_series_equal(out.va_f64, floats, check_names=False)
    tm.assert_series_equal(out.va_i8, ints, check_names=False)
    tm.assert_series_equal(out.va_i16, ints, check_names=False)
    tm.assert_series_equal(out.va_i32, ints, check_names=False)
    tm.assert_series_equal(out.va_i64, ints, check_names=False)

    floats = pd.Series([[x[0] - 1, x[1] - 2, x[2] - 3] for x in FLOAT_VECTORS])
    ints = pd.Series([[x[0] - 1, x[1] - 2, x[2] - 3] for x in INT_VECTORS])

    tm.assert_series_equal(out.va_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_f32_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_f64_lit, floats, check_names=False)
    tm.assert_series_equal(out.va_i8_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i16_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i32_lit, ints, check_names=False)
    tm.assert_series_equal(out.va_i64_lit, ints, check_names=False)


def test_vector_subvector(con: Any) -> None:
    tbl = make_vectors(con.table('datatypes'))

    out = tbl.mutate(
        vs=lambda x: x.vec1.vector_subvector(1, 2).json_array_unpack(),
        vs_f32=lambda x: x.vec1_f32.vector_subvector_f32(1, 2).json_array_unpack_f32(),
        vs_f64=lambda x: x.vec1_f64.vector_subvector_f64(1, 2).json_array_unpack_f64(),
        vs_i8=lambda x: x.vec1_i8.vector_subvector_i8(1, 2).json_array_unpack_i8(),
        vs_i16=lambda x: x.vec1_i16.vector_subvector_i16(1, 2).json_array_unpack_i16(),
        vs_i32=lambda x: x.vec1_i32.vector_subvector_i32(1, 2).json_array_unpack_i32(),
        vs_i64=lambda x: x.vec1_i64.vector_subvector_i64(1, 2).json_array_unpack_i64(),
    ).order_by('id').execute()

    out = out[out.text_vector.notnull()].reset_index(drop=True)

    floats = pd.Series([
        [2.5, -6.69999981],
        [-3.29999995, 7.80000019],
        [6.5999999, 7.9000001],
    ])
    ints = pd.Series([[3, -7], [-3, 8], [7, 8]])

    tm.assert_series_equal(out.vs, floats, check_names=False)
    tm.assert_series_equal(out.vs_f32, floats, check_names=False)
    tm.assert_series_equal(out.vs_f64, floats, check_names=False)
    tm.assert_series_equal(out.vs_i8, ints, check_names=False)
    tm.assert_series_equal(out.vs_i16, ints, check_names=False)
    tm.assert_series_equal(out.vs_i32, ints, check_names=False)
    tm.assert_series_equal(out.vs_i64, ints, check_names=False)


# def test_vector_sum(con: Any) -> None:
#    tbl = make_vectors(con.table('datatypes'))
#
#    out = tbl.mutate(
#        vs=lambda x: x.vec1.vector_sum(),
#        vs_f32=lambda x: x.vec1_f32.vector_sum_f32(),
#        vs_f64=lambda x: x.vec1_f64.vector_sum_f64(),
#        vs_i8=lambda x: x.vec1_i8.vector_sum_i8(),
#        vs_i16=lambda x: x.vec1_i16.vector_sum_i16(),
#        vs_i32=lambda x: x.vec1_i32.vector_sum_i32(),
#        vs_i64=lambda x: x.vec1_i64.vector_sum_i64(),
#    ).order_by('id').execute()
#
#    out = out[out.text_vector.notnull()].reset_index(drop=True)
#
#    print(out)
#
#    floats = pd.Series([[0.0100000007, 6.25, 44.8899956],
#                        [106.090004, 10.8899994, 60.840004],
#                        [5.28999996, 43.5599976, 62.4099998]])
#    ints = pd.Series([[0, 9, 49], [100, 9, 64], [4, 49, 64]])
#
#    tm.assert_series_equal(out.vs, floats, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_f32, floats, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_f64, floats, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_i8, ints, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_i16, ints, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_i32, ints, check_names=False, check_dtype=False)
#    tm.assert_series_equal(out.vs_i64, ints, check_names=False, check_dtype=False)
