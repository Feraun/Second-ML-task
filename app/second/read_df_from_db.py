import time
import pandas as pd
from db import engine
from loguru import logger

def read_df_from_db():

    start = time.time()

    df = pd.read_sql_table('products', con=engine)

    duration = round(time.time() - start, 2)

    logger.info(
        f"{len(df)} products saved to dataframe in {duration} seconds"
    )

    return df