from datetime import time
from predict_price_by_linear_regression import predict_price_by_linear_regression
from read_df_from_db import read_df_from_db
from predict_price_by_catboost import predict_price_by_catboost

def run_predict_job():
    start = time.time()

    df = read_df_from_db()

