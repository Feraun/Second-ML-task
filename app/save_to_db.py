import time

from loguru import logger
from sqlalchemy import create_engine
from config import *

def save_to_db(df):

    start = time.time()

    engine = create_engine(
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    df.to_sql(
        "products",
        engine,
        if_exists="append",
        index=False
    )

    duration = round(time.time() - start, 2)

    logger.info(
        f"{len(df)} products saved to database in {duration} seconds"
    )

    return df
