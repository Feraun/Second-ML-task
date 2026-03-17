from datetime import datetime
from pathlib import Path

from app.db.models_table.save_model_info_to_db import save_model_info_to_db

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

def save_catboost_model(model, metrics, s3service):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    file_path = MODELS_DIR / f"catboost_{now}.cbm"

    model.save_model(str(file_path))

    s3service.upload_file(str(file_path), file_path.name)

    save_model_info_to_db(
        file_name=file_path.name,
        model_type="CBM",
        metrics={"RMSE": metrics["RMSE"],
                 "MAPE": metrics["MAPE"],
                 "R2": metrics["R2"]
                 },
    )