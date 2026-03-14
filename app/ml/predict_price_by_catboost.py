from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

from app.minio.save_catboost_model import save_catboost_model
from minio_service import S3BucketService
from save_model_info_to_db import save_model_info_to_db

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"

def predict_price_by_catboost(x_train, x_test, y_train, y_test, configs, s3service: S3BucketService):

    categorical = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    x_train[categorical] = x_train[categorical].fillna("unknown")
    x_test[categorical] = x_test[categorical].fillna("unknown")

    MODELS_DIR.mkdir(exist_ok=True)

    results = []
    models = []

    for cfg in configs:
        model = CatBoostRegressor(
            iterations=cfg["iterations"],
            depth=cfg["depth"],
            learning_rate=cfg["learning_rate"],
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=False
        )

        model.fit(x_train, y_train, cat_features=categorical)

        pred = model.predict(x_test)

        models.append(model)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        r2 = model.score(x_test, y_test)

        mape = mean_absolute_percentage_error(y_test, pred)

        metrics = {"RMSE": rmse, "MAPE": mape, "R2": r2}

        results.append({
            "iterations": cfg["iterations"],
            "depth": cfg["depth"],
            "learning_rate": cfg["learning_rate"],
            "metrics": {
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            }
        })

        save_catboost_model(model, metrics, s3service)


    results_df = pd.DataFrame(results)

    results_df.to_excel("model_configs.xlsx", index=False)

    return [results, models]