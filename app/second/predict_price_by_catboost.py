from datetime import datetime
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from minio import S3BucketService


def predict_price_by_catboost(x_train, x_test, y_train, y_test, configs, s3service: S3BucketService):

    categorical = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    x_train[categorical] = x_train[categorical].fillna("unknown")
    x_test[categorical] = x_test[categorical].fillna("unknown")

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

        model.save_model(f'catboost_{now}.cbm')

        s3service.upload_file(f'catboost_{now}.cbm', f"catboost_{now}.cbm")


    results_df = pd.DataFrame(results)

    results_df.to_excel("model_configs.xlsx", index=False)

    return results