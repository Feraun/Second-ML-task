from typing import Optional, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from fastAPI.load_models import load_models
from fastAPI.predict_price import predict_prices
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
    vendor: Optional[str] = None
    vendorCode: Optional[str] = None
    categoryId: Optional[int] = None
    price: Optional[float] = None
    params: Optional[Dict[str, str]] = None

class PredictRequest(BaseModel):
    products: List[Product]
    use_catboost: bool = False


@app.on_event("startup")
def startup():
    load_models(service)

@app.post("/predict")
def predict(req: PredictRequest):

    preds = predict_prices(
        req.products,
        req.use_catboost
    )

    return {
        "predictions": preds
    }