import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor


@contextmanager
def get_connection(dsn: Optional[str] = None):
    """
    Obtiene conexión a la base de datos.

    Args:
        dsn: URL de conexión opcional. Si no se proporciona, usa DATABASE_URL del environment.
    """
    connection_string = dsn or os.getenv("DATABASE_URL")
    if not connection_string:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(connection_string)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_all(query, params=None, dsn: Optional[str] = None):
    with get_connection(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or {})
            return cur.fetchall()


def fetch_one(query, params=None, dsn: Optional[str] = None):
    with get_connection(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or {})
            return cur.fetchone()


def execute(query, params=None, dsn: Optional[str] = None):
    with get_connection(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.rowcount


def execute_many(query, params_list, dsn: Optional[str] = None):
    with get_connection(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params_list)
            return cur.rowcount
