import time
from typing import List, Dict

import pandas as pd
import xml.etree.ElementTree as ET

from loguru import logger


def load_products_from_feed(xml_file: str) -> pd.DataFrame:

    start = time.time()

    tree = ET.parse(xml_file)
    root = tree.getroot()
    offers = root.findall(".//offer")

    products: List[Dict] = []

    for offer in offers:
        params: Dict[str, str] = {}
        for p in offer.findall("param"):
            name = p.get("name")
            value = (p.text or "").strip()
            if name:
                params[name] = value

        product = {
            "price": float(offer.findtext("price")),
            "categoryId": int(offer.findtext("categoryId")),
            "params": params
        }
        products.append(product)

    df = pd.DataFrame(products)
    df = df.join(df["params"].apply(pd.Series))
    df = df.drop(columns=["params"], errors="ignore")

    duration = round(time.time() - start, 2)

    logger.info(f"Dataframe was created in {duration} seconds")

    return df

#load_products_from_feed("feed.xml")