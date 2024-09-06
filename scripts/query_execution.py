from db_connection import create_connection

def execute_query(query):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows