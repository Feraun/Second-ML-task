import time
from typing import List

from app.ml.split_df_train_and_test import split_df_train_and_test
from app.minio.save_datasets import save_datasets
from app.minio.save_linear_regression_model import save_linear_regression_model
from config import *
from loguru import logger
from minio_service import S3BucketService
from app.ml.predict_price_by_catboost import predict_price_by_catboost
from app.ml.predict_price_by_linear_regression import predict_price_by_linear_regression
from app.db.read_df_from_db import read_df_from_db

def run_predict_job(configs: List):

    service = S3BucketService(
        MINIO_BUCKET_NAME,
        MINIO_ENDPOINT,
        MINIO_ACCESS_KEY,
        MINIO_SECRET_KEY
    )

    start = time.time()

    df = read_df_from_db()

    x_train, x_test, y_train, y_test = split_df_train_and_test(df)

    pred_by_LR = predict_price_by_linear_regression(
        x_train, x_test, y_train, y_test
    )

    pred_by_cat = predict_price_by_catboost(
        x_train, x_test, y_train, y_test,
        configs,
        service
    )

    duration = round(time.time() - start, 2)

    save_linear_regression_model(pred_by_LR[1], service)
    save_datasets(x_train, x_test, pred_by_LR[1], service)

    logger.info(f"Job was done in {duration} seconds")

    print("LINEAR REGRESSION: ", pred_by_LR[0])
    print("CATBOOST REGRESSION: ", pred_by_cat)