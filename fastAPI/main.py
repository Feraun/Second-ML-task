from fastapi import FastAPI
from app.jobs.download_and_load_df_to_db_job import  run_feed_job

app = FastAPI()

@app.post("/run-feed-job")
def run_job():
    run_feed_job()
    return {"status": "feed job started"}