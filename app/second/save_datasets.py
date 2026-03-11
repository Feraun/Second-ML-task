from datetime import datetime
import time
import numpy as np
from scipy import sparse
from loguru import logger


def save_datasets(x_train, x_test, pipeline, s3service):
    start = time.time()

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    x_train_transformed = pipeline.named_steps["features"].transform(x_train)
    x_test_transformed = pipeline.named_steps["features"].transform(x_test)

    train_csr = sparse.csr_matrix(
        x_train_transformed
    )

    test_csr = sparse.csr_matrix(
        x_test_transformed
    )

    np.savez(
        f"train_dataset_csr__{now}.npz",
        data=train_csr.data,
        indices=train_csr.indices,
        indptr=train_csr.indptr,
        shape=train_csr.shape
    )

    np.savez(
        f"test_dataset_csr__{now}.npz",
        data=test_csr.data,
        indices=test_csr.indices,
        indptr=test_csr.indptr,
        shape=test_csr.shape
    )

    s3service.upload_file(f"train_dataset_csr__{now}.npz", f"train_dataset_csr__{now}.npz")
    s3service.upload_file(f"test_dataset_csr__{now}.npz", f"test_dataset_csr__{now}.npz")

    x_train.to_pickle(f"train_dataset_csr__{now}.pkl")
    x_test.to_pickle(f"test_dataset_csr__{now}.pkl")

    s3service.upload_file(f"train_dataset_csr__{now}.pkl", f"train_dataset_csr__{now}.pkl")
    s3service.upload_file(f"test_dataset_csr__{now}.pkl", f"test_dataset_csr__{now}.pkl")

    duration = round(time.time() - start, 2)

    return logger.info(f"Datasets were saved to minio in {duration} seconds.")