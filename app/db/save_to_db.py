import time
from loguru import logger
from db_engine import engine

def save_to_db(df):

    start = time.time()

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
