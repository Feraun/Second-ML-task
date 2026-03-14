from fastapi import APIRouter

from app.jobs.download_and_load_df_to_db_job import run_feed_job
from app.jobs.predict_job import run_predict_job

router = APIRouter()


@router.post("/train")
def train(data: dict):
    #run_feed_job()
    run_predict_job(data["configs"])
    return {"status": "ok"}