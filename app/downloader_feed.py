import time
import requests
from config import FEED_URL
from loguru import logger

def download_feed():

    #кладем текущее время
    start = time.time()

    r = requests.get(FEED_URL)

    with open("feed.xml", "wb") as f:
        f.write(r.content)

    #вычитаем прошлое взятое время из текущего
    duration = round(time.time() - start, 2)

    logger.info(f"Feed file downloaded in {duration} seconds")

    return "feed.xml"

#download_feed()