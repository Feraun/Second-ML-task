from app.db.models_table.get_last_model import get_last_model

regression_model = None
catboost_model = None

def load_models(service):

    global regression_model
    global catboost_model

    reg_path = get_last_model("LR")
    cat_path = get_last_model("CBM")

    if reg_path:
        regression_model = service.load_model_from_s3(
            reg_path,
            model_type="LR"
        )

    if cat_path:
        catboost_model = service.load_model_from_s3(
            cat_path,
            model_type="CBM"
        )
