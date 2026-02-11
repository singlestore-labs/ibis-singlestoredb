> **⚠️ This project is no longer maintained.**
>
> As of Ibis v12, SingleStoreDB is now officially part of the [Ibis framework](https://ibis-project.org). Please use `ibis-framework` directly instead of this package. See the [Ibis SingleStoreDB documentation](https://ibis-project.org/backends/singlestoredb) for more information.

# <img src="https://raw.githubusercontent.com/singlestore-labs/singlestoredb-python/main/resources/singlestore-logo.png" height="60" valign="middle"/> Ibis backend for SingleStoreDB

## Why Ibis?

[Ibis](https://ibis-project.org) is a Python library that provides a pandas-like API for data manipulation, but instead of operating on in-memory data, it generates SQL queries that run directly in your database. This gives you the best of both worlds: the familiar, expressive syntax of pandas with the performance and scalability of SingleStoreDB.

**Key benefits of using Ibis with SingleStoreDB:**

- **Work with data that doesn't fit in memory** - Your data stays in SingleStoreDB while you build queries using Python. Only the final results are pulled into memory when you call `.execute()`.

- **Lazy evaluation** - Operations are not executed immediately. Instead, Ibis builds up an expression tree that gets compiled into optimized SQL. This allows the database to handle filtering, aggregation, and joins efficiently.

- **Portable code** - Ibis supports multiple database backends. Code written for SingleStoreDB can often be adapted to work with other databases with minimal changes.

- **Type safety** - Ibis validates your expressions before execution, catching errors early rather than at query time.

## Features

- **DataFrame-like API** - Query SingleStoreDB using familiar pandas-style operations
- **Multiple connection methods** - Connect via MySQL protocol or HTTP API
- **Vector functions** - Built-in support for vector operations (dot product, euclidean distance, etc.) for AI/ML workloads
- **JSON functions** - Comprehensive JSON manipulation capabilities for semi-structured data
- **Server accessors** - Direct access to server variables and SHOW commands

## Requirements

- Python 3.8, 3.9, or 3.10
- SingleStoreDB instance (self-hosted or [SingleStore Helios](https://www.singlestore.com/cloud-trial/))

## Install

This package can be installed from PyPI using `pip`:
```
pip install ibis-singlestoredb
```

## Usage

### Connection Methods

Connections to the SingleStoreDB database can be established in several ways:

**Connection string (MySQL protocol)**
```python
import ibis

conn = ibis.singlestoredb.connect("user:password@host:3306/db_name")
```

**Connection string (HTTP API)**
```python
conn = ibis.singlestoredb.connect("http://user:password@host:8080/db_name")
```

**Environment variable**
```python
# Set SINGLESTOREDB_URL environment variable first
# export SINGLESTOREDB_URL="user:password@host:3306/db_name"

conn = ibis.singlestoredb.connect()
```

**Keyword arguments**
```python
conn = ibis.singlestoredb.connect(
    host="localhost",
    port=3306,
    user="user",
    password="password",
    database="db_name",
    driver="mysql",  # or "http"
)
```

### DataFrame Operations

**Get a table reference**
```python
# Get an Ibis table expression
employees = conn.table("employees")
```

**Filtering and selecting**
```python
# Filter rows and select columns
result = employees[employees.salary > 50000][["name", "department", "salary"]]

# Alternative syntax
result = employees.filter(employees.salary > 50000).select("name", "department", "salary")
```

**Grouping and aggregation**
```python
# Group by department and calculate statistics
stats = (
    employees
    .group_by("department")
    .aggregate(
        avg_salary=employees.salary.mean(),
        employee_count=employees.count(),
    )
)
```

**Execute and retrieve results**
```python
# Execute and get a pandas DataFrame
df = result.execute()
```

**Interactive mode**
```python
# Enable interactive mode for automatic execution
ibis.options.interactive = True

# Now expressions are automatically executed when displayed
employees.head()
```

**Creating tables from pandas DataFrames**
```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 87.3, 92.1],
})

# Create a table in SingleStoreDB
conn.create_table("scores", df)

# With schema overrides
conn.create_table(
    "scores",
    df,
    schema_overrides={"score": "float32"},
)

# With storage type specification
conn.create_table("scores", df, storage_type="rowstore")
```

### Vector Functions

SingleStoreDB provides native vector operations for AI/ML workloads. These functions are available as methods on binary columns containing packed vectors.

**Packing and unpacking vectors**
```python
# Pack a JSON array into a binary vector (32-bit floats)
packed = ibis.literal("[1.0, 2.0, 3.0]").json_array_pack()

# Unpack a binary vector back to JSON
unpacked = vector_column.json_array_unpack()
```

**Vector similarity**
```python
# Dot product between two vectors
similarity = vector_col.dot_product([1.0, 2.0, 3.0])

# Euclidean distance
distance = vector_col.euclidean_distance([1.0, 2.0, 3.0])
```

**Vector arithmetic**
```python
# Element-wise operations
result = vector_a.vector_add(vector_b)
result = vector_a.vector_sub(vector_b)
result = vector_a.vector_mul(vector_b)

# Scalar multiplication
result = vector_col.scalar_vector_mul(2.0)
```

**Vector utilities**
```python
# Get number of elements
num_elements = vector_col.vector_num_elements()

# Get sum of elements
total = vector_col.vector_elements_sum()

# Extract element at index
element = vector_col.vector_kth_element(0)

# Sort vector elements
sorted_vec = vector_col.vector_sort()
```

### JSON Functions

SingleStoreDB provides comprehensive JSON manipulation functions.

**Extracting values**
```python
# Extract a string value
name = json_col.json_extract_string("name")

# Extract a numeric value
price = json_col.json_extract_double("price")

# Extract a nested value using path
city = json_col.json_extract_string("address", "city")

# Extract an integer
count = json_col.json_extract_bigint("count")

# Extract nested JSON
nested = json_col.json_extract_json("metadata")
```

**Querying JSON structure**
```python
# Get keys from a JSON object
keys = json_col.json_keys()

# Get the length of a JSON array or object
length = json_col.json_length()

# Get the type of a JSON value
json_type = json_col.json_get_type()
```

**Modifying JSON**
```python
# Delete a key from a JSON object
modified = json_col.json_delete_key("unwanted_field")

# Set a value in a JSON object
updated = json_col.json_set_string("status", "active")
```

### Server Accessors

The connection object provides direct access to server features.

**SHOW commands**
```python
# Access SHOW command results
tables = conn.show.tables
databases = conn.show.databases
status = conn.show.status
```

**Server variables**
```python
# Access global server variables
global_vars = conn.globals

# Access session-local variables
local_vars = conn.locals

# Access combined variables
all_vars = conn.vars
```

## Examples

There are some example Jupyter notebooks in the
[examples](https://github.com/singlestore-labs/ibis-singlestoredb/tree/main/examples)
directory.


## License

This library is licensed under the [Apache 2.0 License](https://raw.githubusercontent.com/singlestore-labs/singlestoredb-python/main/LICENSE?token=GHSAT0AAAAAABMGV6QPNR6N23BVICDYK5LAYTVK5EA).

## Resources

* [SingleStore](https://singlestore.com)
* [Ibis](https://ibis-project.org)
* [Python](https://python.org)

## User agreement

SINGLESTORE, INC. ("SINGLESTORE") AGREES TO GRANT YOU AND YOUR COMPANY ACCESS TO THIS OPEN SOURCE SOFTWARE CONNECTOR ONLY IF (A) YOU AND YOUR COMPANY REPRESENT AND WARRANT THAT YOU, ON BEHALF OF YOUR COMPANY, HAVE THE AUTHORITY TO LEGALLY BIND YOUR COMPANY AND (B) YOU, ON BEHALF OF YOUR COMPANY ACCEPT AND AGREE TO BE BOUND BY ALL OF THE OPEN SOURCE TERMS AND CONDITIONS APPLICABLE TO THIS OPEN SOURCE CONNECTOR AS SET FORTH BELOW (THIS "AGREEMENT"), WHICH SHALL BE DEFINITIVELY EVIDENCED BY ANY ONE OF THE FOLLOWING MEANS: YOU, ON BEHALF OF YOUR COMPANY, CLICKING THE "DOWNLOAD, "ACCEPTANCE" OR "CONTINUE" BUTTON, AS APPLICABLE OR COMPANY'S INSTALLATION, ACCESS OR USE OF THE OPEN SOURCE CONNECTOR AND SHALL BE EFFECTIVE ON THE EARLIER OF THE DATE ON WHICH THE DOWNLOAD, ACCESS, COPY OR INSTALL OF THE CONNECTOR OR USE ANY SERVICES (INCLUDING ANY UPDATES OR UPGRADES) PROVIDED BY SINGLESTORE.
BETA SOFTWARE CONNECTOR

Customer Understands and agrees that it is  being granted access to pre-release or "beta" versions of SingleStore's open source software connector ("Beta Software Connector") for the limited purposes of non-production testing and evaluation of such Beta Software Connector. Customer acknowledges that SingleStore shall have no obligation to release a generally available version of such Beta Software Connector or to provide support or warranty for such versions of the Beta Software Connector  for any production or non-evaluation use.

NOTWITHSTANDING ANYTHING TO THE CONTRARY IN ANY DOCUMENTATION,  AGREEMENT OR IN ANY ORDER DOCUMENT, SINGLESTORE WILL HAVE NO WARRANTY, INDEMNITY, SUPPORT, OR SERVICE LEVEL, OBLIGATIONS WITH
RESPECT TO THIS BETA SOFTWARE CONNECTOR (INCLUDING TOOLS AND UTILITIES).

APPLICABLE OPEN SOURCE LICENSE: Apache 2.0

IF YOU OR YOUR COMPANY DO NOT AGREE TO THESE TERMS AND CONDITIONS, DO NOT CHECK THE ACCEPTANCE BOX, AND DO NOT DOWNLOAD, ACCESS, COPY, INSTALL OR USE THE SOFTWARE OR THE SERVICES.
