# <img src="resources/singlestore-logo.png" height="60" valign="middle"/> Ibis backend for SingleStoreDB

This project contains an [Ibis](https://ibis-project.org) backend
for the SingleStore database. This allows you to access and manipulate your
data using a DataFrame-like API.

## Install

This package can be installed from PyPI using `pip`:
```
pip install ibis-singlestoredb
```

## Usage

Connections to the SingleStoreDB database are made using URLs that specify
the connection driver package, server hostname, server port, and user
credentials.
```
import ibis

# Connect using the default connector
conn = ibis.singlestoredb.connect('user:password@host:3306/db_name')

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
conn = ibis.singlestoredb.connect('http://user:password@host:8080/db_name')
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

SINGLESTORE, INC. ("SINGLESTORE") AGREES TO GRANT YOU AND YOUR COMPANY ACCESS TO THIS OPEN SOURCE SOFTWARE CONNECTOR ONLY IF (A) YOU AND YOUR COMPANY REPRESENT AND WARRANT THAT YOU, ON BEHALF OF YOUR COMPANY, HAVE THE AUTHORITY TO LEGALLY BIND YOUR COMPANY AND (B) YOU, ON BEHALF OF YOUR COMPANY ACCEPT AND AGREE TO BE BOUND BY ALL OF THE OPEN SOURCE TERMS AND CONDITIONS APPLICABLE TO THIS OPEN SOURCE CONNECTOR AS SET FORTH BELOW (THIS “AGREEMENT”), WHICH SHALL BE DEFINITIVELY EVIDENCED BY ANY ONE OF THE FOLLOWING MEANS: YOU, ON BEHALF OF YOUR COMPANY, CLICKING THE “DOWNLOAD, “ACCEPTANCE” OR “CONTINUE” BUTTON, AS APPLICABLE OR COMPANY’S INSTALLATION, ACCESS OR USE OF THE OPEN SOURCE CONNECTOR AND SHALL BE EFFECTIVE ON THE EARLIER OF THE DATE ON WHICH THE DOWNLOAD, ACCESS, COPY OR INSTALL OF THE CONNECTOR OR USE ANY SERVICES (INCLUDING ANY UPDATES OR UPGRADES) PROVIDED BY SINGLESTORE.
BETA SOFTWARE CONNECTOR

Customer Understands and agrees that it is  being granted access to pre-release or “beta” versions of SingleStore’s open source software connector (“Beta Software Connector”) for the limited purposes of non-production testing and evaluation of such Beta Software Connector. Customer acknowledges that SingleStore shall have no obligation to release a generally available version of such Beta Software Connector or to provide support or warranty for such versions of the Beta Software Connector  for any production or non-evaluation use.

NOTWITHSTANDING ANYTHING TO THE CONTRARY IN ANY DOCUMENTATION,  AGREEMENT OR IN ANY ORDER DOCUMENT, SINGLESTORE WILL HAVE NO WARRANTY, INDEMNITY, SUPPORT, OR SERVICE LEVEL, OBLIGATIONS WITH
RESPECT TO THIS BETA SOFTWARE CONNECTOR (INCLUDING TOOLS AND UTILITIES).

APPLICABLE OPEN SOURCE LICENSE: Apache 2.0

IF YOU OR YOUR COMPANY DO NOT AGREE TO THESE TERMS AND CONDITIONS, DO NOT CHECK THE ACCEPTANCE BOX, AND DO NOT DOWNLOAD, ACCESS, COPY, INSTALL OR USE THE SOFTWARE OR THE SERVICES.
