
create table if not exists datatypes (
    id int unsigned,
    tinyint_c tinyint,
    smallint_c smallint,
    mediumint_c mediumint,
    int_c int,
    bigint_c bigint,
    float_c float,
    double_c double,
    decimal_c decimal(18,6),
    decimal5_c decimal(18,5),
    numeric_c numeric,
    date_c date,
    time_c time,
    time6_c time(6),
    datetime_c datetime,
    datetime6_c datetime(6),
    timestamp_c timestamp,
    timestamp6_c timestamp(6),
    char32_c char(32),
    varchar42_c varchar(42),
    longtext_c longtext,
    mediumtext_c mediumtext,
    tinytext_c tinytext,
    text_c text,
    text4_c text(4),
    blob_c blob,
    enum_sml_c enum('small', 'medium', 'large'),
    set_abcd_c set('a', 'b', 'c', 'd'),
    negative_int_c int,
    negative_float_c float,
    bool_c bool,

    -- Complex types
    text_vector text,
    json_vector json,
    text_list text,
    json_list json,
    text_obj text,
    json_obj json
);

load data local infile '{{TEST_PATH}}/datatypes.csv' into table datatypes
    columns terminated by ',' optionally enclosed by "'" lines terminated by '\n'
    ignore 1 lines;
