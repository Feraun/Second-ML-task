from db_engine import engine

from create_models_table import models_table

from datetime import datetime

def save_model_info_to_db(file_name, model_type, metrics):

    with engine.begin() as conn:
        conn.execute(
            models_table.insert().values(
                fileName=file_name,
                type=model_type,
                metrics=metrics,
                dateTime=datetime.now(),
            )
        )