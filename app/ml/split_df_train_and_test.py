from sklearn.model_selection import train_test_split


def split_df_train_and_test(df):

    x = df.drop(columns=["price"])
    y = df["price"]

    x.columns = [str(c) for c in x.columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.4,
        random_state=42
    )

    return x_train, x_test, y_train, y_test