import pandas as pd


def prepare_df(products, model):

    rows = []

    for p in products:
        params = p.params

        row = {}

        if params:
            row.update(params)

        rows.append(row)

    df = pd.DataFrame(rows)

    features = None

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)

    if hasattr(model, "feature_names_"):
        features = list(model.feature_names_)

    if features is not None:
        for col in features:
            if col not in df.columns:
                df[col] = None

        df = df[features]

    df = df.fillna(0)

    return df