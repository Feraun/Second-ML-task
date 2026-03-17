import pandas as pd
import fastAPI.load_models as lm


def predict_prices(products, use_catboost: bool):
    rows = []

    for p in products:

        row = {
            "id": p.id,
            "name": p.name,
            "vendor": p.vendor,
            "categoryId": p.categoryId,
        }

        if p.params:
            row.update(p.params)

        rows.append(row)

    data = pd.DataFrame(rows)

    model = lm.catboost_model if use_catboost else lm.regression_model

    preds = model.predict(data)

    return preds.tolist()