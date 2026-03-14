import time
from app.utils.downloader_feed import download_feed
from app.utils.parser_feed import load_products_from_feed
from app.db.save_to_db import save_to_db
from loguru import logger

def run_feed_job():

    start = time.time()

    file_path = download_feed()

    df = load_products_from_feed(file_path)

    save_to_db(df)

    duration = round(time.time() - start, 2)

    logger.info(f"Feed job completed in {duration} seconds")