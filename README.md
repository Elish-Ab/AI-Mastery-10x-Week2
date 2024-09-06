# Project: WEEK2

## Overview
This project is focused on interacting with a PostgreSQL database to perform data extraction, loading, and analysis. It includes Jupyter Notebooks for data loading, as well as Python scripts for handling database connections, data extraction, loading, and analysis.

## Project Structure
- **notebooks/**: Contains Jupyter Notebooks for interacting with the PostgreSQL database.
- **scripts/**: Contains Python scripts for database connections, data extraction, loading, and analysis.
- **src/**: 
- **tests/**: test code
- **.env**: Configuration file containing environment variables such as database credentials.

## Environment Configuration
The project requires a `.env` file in the root directory with the following configuration:

```plaintext
DB_HOST=<database_host>
DB_PORT=<database_port>
DB_NAME=<database_name>
DB_USER=<database_user>
DB_PASSWORD=<database_password>
