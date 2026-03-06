from fastapi import FastAPI
from app.job import  run_feed_job

app = FastAPI()

@app.post("/run-feed-job")
def run_job():
    run_feed_job()
    return {"status": "feed job started"}