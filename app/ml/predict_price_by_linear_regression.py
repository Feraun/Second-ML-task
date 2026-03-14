from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

from app.minio.save_linear_regression_model import save_linear_regression_model
from minio_service import S3BucketService


def predict_price_by_linear_regression(
    x_train,
    x_test,
    y_train,
    y_test,
    s3service: S3BucketService):

    #делаем столбец числовым
    x_train["Суммарная мощность"] = x_train["Суммарная мощность"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    x_test["Суммарная мощность"] = x_test["Суммарная мощность"].str.replace(r"[^\d.]", "", regex=True).astype(float)

    #вытаскиваем категориальные и числовые столбцы, как Series
    categorical = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = x_train.select_dtypes(exclude=["object", "string"]).columns.tolist()

    # print("categorical:", categorical)
    # print("numeric:", numeric)

    #обработчик Series'ов
    preprocessor = make_column_transformer(

(Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot-encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical),

        (Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", MinMaxScaler())
        ]),
        numeric)
    )

    lr_pipeline = Pipeline([
        ("features", preprocessor),
        ("model", LinearRegression())
    ])

    # обучение
    # трансформер получает x_train, делает fit и возвращает преобразованный x.
    # модель получает x и y_train, делает fit — обучает модель.
    lr_pipeline.fit(x_train, y_train)

    # уже обученная модель предсказывает цены
    # предсказанный массив цен
    lr_pred = lr_pipeline.predict(x_test)

    metrics = {"RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
               "R2": r2_score(y_test, lr_pred),
               "MAPE": mean_absolute_percentage_error(y_test, lr_pred)}

    save_linear_regression_model(lr_pipeline, s3service, metrics)

    return [metrics, lr_pipeline]

# df = read_df_from_db()
# x_train, x_test, y_train, y_test = difference_df_train_and_test(df)
# pred_by_LR = predict_price_by_linear_regression(x_train, x_test, y_train, y_test)
# print(pred_by_LR[0])