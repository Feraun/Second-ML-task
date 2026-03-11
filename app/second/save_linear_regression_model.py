import pickle
from datetime import datetime
from minio import S3BucketService

def save_linear_regression_model(pipeline, s3service: S3BucketService):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(f"regression__{now}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    s3service.upload_file(f"regression__{now}.pkl", f"regression__{now}.pkl")