# <img src="resources/singlestore-logo.png" height="60" valign="middle"/> Ibis backend for SingleStore

This project contains an [Ibis](https://ibis-project.org) backend
for the SingleStore database. This allows you to access and manipulate your
data using a DataFrame-like API.

## Install

This package can be install from PyPI using `pip`:
```
pip install ibis-singlestore
```

## Usage

Connections to the SingleStore database are made using URLs that specify
the connection driver package, server hostname, server port, and user
credentials.
```
import ibis

# Connect using the default connector
conn = ibis.singlestore.connect('user:password@host:3306/db_name')

# Get an Ibis DataFrame object that points to the 'employees' table
employees = conn.table('employees')

# Apply DataFrame-like operations (lazily)
res = employees[employees.name.like('Smith')][employees.deptId > 1]

# Execute the operations
res.execute()
```

Connecting to the HTTP API is done as follows:
```
# Use the HTTP API connector
conn = ibis.singlestore.connect('http://user:password@host:8080/db_name')
```


## Resources

* [SingleStore](https://singlestore.com)
* [Ibis](https://ibis-project.org)
* [Python](https://python.org)
