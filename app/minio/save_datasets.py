from datetime import datetime
import time
import numpy as np
from scipy import sparse
from loguru import logger
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS_DIR = BASE_DIR / "datasets"
NPZ_DIR = DATASETS_DIR / "npz"
PKL_DIR = DATASETS_DIR / "pkl"


def save_datasets(x_train, x_test, pipeline, s3service):

    start = time.time()

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    x_train_transformed = pipeline.named_steps["features"].transform(x_train)
    x_test_transformed = pipeline.named_steps["features"].transform(x_test)

    train_csr = sparse.csr_matrix(x_train_transformed)
    test_csr = sparse.csr_matrix(x_test_transformed)

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    train_csr_path = NPZ_DIR / f"train_dataset_csr__{now}.npz"
    test_csr_path = NPZ_DIR / f"test_dataset_csr__{now}.npz"

    np.savez(
        train_csr_path,
        data=train_csr.data,
        indices=train_csr.indices,
        indptr=train_csr.indptr,
        shape=train_csr.shape,
    )

    np.savez(
        test_csr_path,
        data=test_csr.data,
        indices=test_csr.indices,
        indptr=test_csr.indptr,
        shape=test_csr.shape,
    )

    s3service.upload_file(str(train_csr_path), train_csr_path.name)
    s3service.upload_file(str(test_csr_path), test_csr_path.name)

    train_cbm_path = NPZ_DIR / f"train_dataset_cbm__{now}.npz"
    test_cbm_path = NPZ_DIR / f"test_dataset_cbm__{now}.npz"

    np.savez(train_cbm_path, x_train_transformed)
    np.savez(test_cbm_path, x_test_transformed)

    s3service.upload_file(str(train_cbm_path), train_cbm_path.name)
    s3service.upload_file(str(test_cbm_path), test_cbm_path.name)

    train_pkl_path = PKL_DIR / f"train_dataset__{now}.pkl"
    test_pkl_path = PKL_DIR / f"test_dataset__{now}.pkl"

    x_train.to_pickle(train_pkl_path)
    x_test.to_pickle(test_pkl_path)

    s3service.upload_file(str(train_pkl_path), train_pkl_path.name)
    s3service.upload_file(str(test_pkl_path), test_pkl_path.name)

    duration = round(time.time() - start, 2)

    logger.info(f"Datasets saved in {duration} sec")