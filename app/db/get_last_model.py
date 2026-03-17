from sqlalchemy import text
from db_engine import engine

def get_last_model(model_type: str):

    query = text("""
        SELECT "fileName"
        FROM models m
        WHERE "type" = :model_type
        ORDER BY "dateTime" DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"model_type": model_type}).fetchone()

    return result[0] if result else None