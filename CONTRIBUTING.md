# Contributing to ibis-singlestoredb

Thank you for your interest in contributing to ibis-singlestoredb! This document provides guidelines and instructions for contributing.

## Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/singlestore-labs/ibis-singlestoredb.git
cd ibis-singlestoredb
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

4. Install test dependencies:
```bash
uv pip install pytest coverage
```

5. Install pre-commit hooks:
```bash
uv pip install pre-commit
pre-commit install
```

## Pre-commit Checks

This project uses pre-commit hooks to ensure code quality. The following checks are run automatically before each commit:

- **trailing-whitespace** - Removes trailing whitespace
- **end-of-file-fixer** - Ensures files end with a newline
- **check-docstring-first** - Checks that docstrings come first in modules
- **check-json** - Validates JSON files
- **debug-statements** - Checks for debugger imports and breakpoints
- **double-quote-string-fixer** - Converts single quotes to double quotes
- **requirements-txt-fixer** - Sorts requirements files
- **flake8** - Python linting (with flake8-typing-imports)
- **autopep8** - Automatic PEP 8 formatting
- **reorder-python-imports** - Sorts and organizes imports
- **add-trailing-comma** - Adds trailing commas to function calls
- **setup-cfg-fmt** - Formats setup.cfg
- **mypy** - Static type checking

To run pre-commit checks manually on all files:
```bash
pre-commit run --all-files
```

## Running Tests

### Environment Variables

Configure your test environment using the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IBIS_TEST_SINGLESTOREDB_HOST` | `localhost` | Database host |
| `IBIS_TEST_SINGLESTOREDB_PORT` | `9306` | Database port |
| `IBIS_TEST_SINGLESTOREDB_USER` | `ibis` | Database user |
| `IBIS_TEST_SINGLESTOREDB_PASSWORD` | `ibis` | Database password |
| `IBIS_TEST_SINGLESTOREDB_DATABASE` | `ibis_testing` | Database name |
| `IBIS_TEST_SINGLESTOREDB_DRIVER` | `mysql` | Connection driver |

Alternatively, set a single connection URL:
```bash
export SINGLESTOREDB_URL="user:password@host:port/database"
```

### Running Tests

Run the test suite using the MySQL protocol (default, port 3306):
```bash
IBIS_TEST_SINGLESTOREDB_DRIVER=mysql \
IBIS_TEST_SINGLESTOREDB_PORT=3306 \
pytest ibis_singlestoredb/tests/
```

Run the test suite using the HTTP Data API (port 9000):
```bash
IBIS_TEST_SINGLESTOREDB_DRIVER=http \
IBIS_TEST_SINGLESTOREDB_PORT=9000 \
pytest ibis_singlestoredb/tests/
```

Run tests with coverage:
```bash
pytest -v --cov=ibis_singlestoredb ibis_singlestoredb/tests/
```

Run a specific test file:
```bash
pytest ibis_singlestoredb/tests/test_client.py -v
```

## Questions?

If you have questions, please visit the [SingleStore Forums](https://www.singlestore.com/forum/).
