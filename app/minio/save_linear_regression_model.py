import pickle
from datetime import datetime
from minio_service import S3BucketService
from pathlib import Path

from app.db.models_table.save_model_info_to_db import save_model_info_to_db

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

def save_linear_regression_model(pipeline, s3service: S3BucketService, metrics):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    MODELS_DIR.mkdir(exist_ok=True)

    file_path = MODELS_DIR / f"regression__{now}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(pipeline, f)

    s3service.upload_file(str(file_path), file_path.name)

    save_model_info_to_db(
        file_name=file_path.name,
        model_type="LR",
        metrics={"RMSE": metrics["RMSE"],
                 "MAPE": metrics["MAPE"],
                 "R2": metrics["R2"]}
    )