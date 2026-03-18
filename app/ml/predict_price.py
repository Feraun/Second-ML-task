import app.minio.load_models as lm

def predict_prices(products, use_catboost: bool):
    model = lm.catboost_model if use_catboost else lm.regression_model
    preds = model.predict(products)
    return preds.tolist()