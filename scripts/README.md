# Scripts

## Overview
This folder contains Python scripts that handle various operations related to data extraction, loading, and analysis from the PostgreSQL database.

## Scripts

### `db_connection.py`
This script handles the connection between the PostgreSQL database and the Python environment using SQLAlchemy. It is used by other scripts in this folder to establish and manage database connections.

### `extract_data.py`
This script is responsible for extracting data from the PostgreSQL database. It executes SQL queries and retrieves the required data for further processing or analysis.

### `load_data.py`
This script loads data into the PostgreSQL database. It reads data from specified sources and inserts it into the appropriate tables within the database.

### `query_execution.py`
This script is used to select and retrieve data from the PostgreSQL database based on specific queries. It is useful for fetching subsets of data for analysis.

### `user_analysis_script.py`
This script performs Task 1 analysis by processing and analyzing user data retrieved from the PostgreSQL database. It utilizes the data extraction and querying functions provided by the other scripts in this folder.

## Usage
- Ensure the `db_connection.py` script is configured with the correct database credentials.
- Run the scripts in the order specified for data loading, extraction, and analysis.
