from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from minio_service import S3BucketService

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"

def predict_price_by_catboost(x_train, x_test, y_train, y_test, configs, s3service: S3BucketService):

    categorical = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    x_train[categorical] = x_train[categorical].fillna("unknown")
    x_test[categorical] = x_test[categorical].fillna("unknown")

    MODELS_DIR.mkdir(exist_ok=True)

    results = []

    for cfg in configs:
        model = CatBoostRegressor(
            iterations=cfg["iterations"],
            depth=cfg["depth"],
            learning_rate=cfg["learning_rate"],
            loss_function="RMSE",
            verbose=False
        )

        model.fit(x_train, y_train, cat_features=categorical)

        pred = model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "iterations": cfg["iterations"],
            "depth": cfg["depth"],
            "learning_rate": cfg["learning_rate"],
            "RMSE": rmse
        })

        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        file_path = MODELS_DIR / f"catboost_{now}.cbm"

        model.save_model(str(file_path))

        s3service.upload_file(str(file_path), file_path.name)


    results_df = pd.DataFrame(results)

    results_df.to_excel("model_configs.xlsx", index=False)

    return results