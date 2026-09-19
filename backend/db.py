"""
Veritus — Database Management Layer (backend/db.py)

Handles PostgreSQL connections, schema initialization, and CRUD operations
for datasets and hypotheses. Supports both local Docker PostgreSQL and
cloud PostgreSQL (e.g. Supabase).
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv())

def get_db_url() -> str:
    """Constructs or retrieves the database URL from environment variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    
    # Fallback to individual components
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "ai_data_analyst")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

@contextmanager
def get_connection():
    """
    Context manager that yields an active PostgreSQL connection.
    Automatically handles commits and rollbacks on exceptions.
    """
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    """
    Initializes the database schema if tables do not already exist.
    Creates:
      1. datasets: metadata for uploaded CSV files
      2. hypotheses: all generated, tested, and refined hypotheses with test metrics
    """
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS hypotheses (
            id SERIAL PRIMARY KEY,
            dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
            hypothesis_text TEXT NOT NULL,
            parent_hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL,
            test_method TEXT,
            p_value FLOAT,
            effect_size FLOAT,
            verdict TEXT CHECK (verdict IN ('validated', 'rejected', 'inconclusive')),
            explanation TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_hypotheses_dataset ON hypotheses(dataset_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_hypotheses_parent ON hypotheses(parent_hypothesis_id);
        """
    ]
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for statement in ddl_statements:
                cur.execute(statement)
    print("Database schema initialized successfully.")

def create_dataset(name: str) -> int:
    """
    Registers a new dataset record in PostgreSQL.
    Returns:
        int: The auto-generated dataset_id.
    """
    query = """
        INSERT INTO datasets (name)
        VALUES (%s)
        RETURNING id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (name,))
            dataset_id = cur.fetchone()[0]
            return dataset_id

def save_hypothesis(
    dataset_id: int,
    hypothesis_text: str,
    test_method: Optional[str] = None,
    p_value: Optional[float] = None,
    effect_size: Optional[float] = None,
    verdict: Optional[str] = None,
    explanation: Optional[str] = None,
    parent_hypothesis_id: Optional[int] = None
) -> int:
    """
    Saves a hypothesis, its statistical test results, and narrative explanation.
    Returns:
        int: The auto-generated hypothesis_id.
    """
    query = """
        INSERT INTO hypotheses (
            dataset_id,
            hypothesis_text,
            parent_hypothesis_id,
            test_method,
            p_value,
            effect_size,
            verdict,
            explanation
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (
                dataset_id,
                hypothesis_text,
                parent_hypothesis_id,
                test_method,
                p_value,
                effect_size,
                verdict,
                explanation
            ))
            hypothesis_id = cur.fetchone()[0]
            return hypothesis_id

def get_dataset(dataset_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a single dataset by its ID."""
    query = "SELECT id, name, uploaded_at FROM datasets WHERE id = %s;"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (dataset_id,))
            row = cur.fetchone()
            return dict(row) if row else None

def get_hypotheses_by_dataset(dataset_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves all hypotheses associated with a dataset, sorted chronologically.
    Useful for rendering validated findings and lineage trees in Streamlit.
    """
    query = """
        SELECT 
            id,
            dataset_id,
            hypothesis_text,
            parent_hypothesis_id,
            test_method,
            p_value,
            effect_size,
            verdict,
            explanation,
            created_at
        FROM hypotheses
        WHERE dataset_id = %s
        ORDER BY id ASC;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (dataset_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]

def list_datasets() -> List[Dict[str, Any]]:
    """Lists all analyzed datasets with a count of hypotheses generated."""
    query = """
        SELECT 
            d.id,
            d.name,
            d.uploaded_at,
            COUNT(h.id) AS hypothesis_count,
            COUNT(CASE WHEN h.verdict = 'validated' THEN 1 END) AS validated_count,
            COUNT(CASE WHEN h.verdict = 'rejected' THEN 1 END) AS rejected_count
        FROM datasets d
        LEFT JOIN hypotheses h ON d.id = h.dataset_id
        GROUP BY d.id, d.name, d.uploaded_at
        ORDER BY d.uploaded_at DESC;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [dict(row) for row in rows]

if __name__ == "__main__":
    print("--- Testing Veritus Database Layer ---")
    init_db()
    
    # Test insertion
    ds_id = create_dataset("test_sales_data.csv")
    print(f"Created test dataset ID: {ds_id}")
    
    hypo_id = save_hypothesis(
        dataset_id=ds_id,
        hypothesis_text="Higher advertising spend is positively correlated with sales volume.",
        test_method="correlation (Pearson)",
        p_value=0.0024,
        effect_size=0.78,
        verdict="validated",
        explanation="Strong positive correlation (r=0.78, p=0.0024) confirms advertising drives sales volume."
    )
    print(f"Saved test hypothesis ID: {hypo_id}")
    
    hypotheses = get_hypotheses_by_dataset(ds_id)
    print(f"Retrieved {len(hypotheses)} hypotheses for dataset {ds_id}:")
    for h in hypotheses:
        print(f"  - [{h['verdict'].upper()}] {h['hypothesis_text']} (p={h['p_value']}, effect={h['effect_size']})")
    
    datasets = list_datasets()
    print(f"Total datasets logged: {len(datasets)}")
    print("--- Test Passed Successfully! ---")
