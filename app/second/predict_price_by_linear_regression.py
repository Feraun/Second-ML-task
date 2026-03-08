from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def predict_price_by_linear_regression(df):

    #делаем столбец числовым
    df["Суммарная мощность"] = df["Суммарная мощность"].str.replace(r"[^\d.]", "", regex=True).astype(float)

    #заполняем пустые значения
    myImputer = SimpleImputer(strategy='constant', fill_value=0)

    df[["Суммарная мощность"]] = myImputer.fit_transform(df[["Суммарная мощность"]])

    # датафрейм без столбца price
    x = df.drop(columns=["price"])

    # массив из прайсов
    y = df["price"]

    #названия столбцов делаем строками
    x.columns = [str(c) for c in x.columns]

    #разбиваем на дфы и массивы для обучения и тестирования
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    # plt.scatter(x=X_train["Суммарная мощность"], y=y_train)
    # plt.xlabel("Суммарная мощность")
    # plt.ylabel("Цена")
    # plt.show()

    #вытаскиваем категориальные и числовые столбцы, как Series
    categorical = x.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = x.select_dtypes(exclude=["object", "string"]).columns.tolist()

    # print("categorical:", categorical)
    # print("numeric:", numeric)

    #обработчик Series'ов
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), categorical),
        (MinMaxScaler(), numeric)
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

    #корень из средней квадратичной ошибки
    rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print("LinearRegression RMSE:", rmse)