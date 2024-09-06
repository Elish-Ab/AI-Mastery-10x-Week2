# scripts/load_data.py
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from db_connection import create_connection
from db_connection import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def load_data_using_sqlalchemy(query):
    connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)
    df = pd.read_sql_query(query, engine)

    return df

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the database
        conn = create_connection()
        # Load data using pandas
        df = pd.read_sql_query(query, conn)

        # Close the database connection
        conn.close()

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None