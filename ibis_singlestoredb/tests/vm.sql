create table vectors (
    id int,
    text_vector text
);

insert into vectors(id, text_vector) values (1, '[0.1, 2.5, -6.7]');

SELECT id,
       text_vector,
       json_array_unpack(vector_mul(json_array_pack(text_vector), json_array_pack(text_vector))) AS vm,
       json_array_unpack_f32(vector_mul_f32(json_array_pack_f32(text_vector), json_array_pack_f32(text_vector))) AS vm_f32
FROM vectors;

-- +------+------------------+----------+--------------------------------+
-- | id   | text_vector      | vm       | vm_f32                         |
-- +------+------------------+----------+--------------------------------+
-- |    1 | [0.1, 2.5, -6.7] | [0,0,-0] | [0.0100000007,6.25,44.8899956] |
-- +------+------------------+----------+--------------------------------+
