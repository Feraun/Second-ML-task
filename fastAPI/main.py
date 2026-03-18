from typing import Optional, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

import app.minio.load_models as lm
from app.ml.predict_price import predict_prices
from app.utils.prepare_df import prepare_df
from fastAPI.train_api import router
from minio_service import S3BucketService
from config import *
app = FastAPI()

app.include_router(router)

service = S3BucketService(
    MINIO_BUCKET_NAME,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY
)

class Product(BaseModel):
    id: str
    name: str
    categoryId: Optional[int] = None
    price: Optional[float] = None
    params: Optional[Dict[str, str]] = None

class PredictRequest(BaseModel):
    products: List[Product]
    use_catboost: bool = False


@app.post("/predict")
def predict(req: PredictRequest):

    lm.load_models(service)

    df = prepare_df(req.products, lm.catboost_model if req.use_catboost else lm.regression_model)

    print(df)

    preds = predict_prices(
        df,
        req.use_catboost
    )

    return {
        "predictions": preds
    }