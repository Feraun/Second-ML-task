import pickle
from datetime import datetime
from minio_service import S3BucketService
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

def save_linear_regression_model(pipeline, s3service: S3BucketService):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    MODELS_DIR.mkdir(exist_ok=True)

    file_path = MODELS_DIR / f"regression__{now}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(pipeline, f)

    s3service.upload_file(str(file_path), file_path.name)