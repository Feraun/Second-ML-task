import requests

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

API_URL = "http://localhost:8000/train"

def call_train_api(configs):

    try:
        r = requests.post(API_URL, json={"configs": configs})

        print("API status:", r.status_code)

    except Exception as e:
        print("Scheduler error:", e)

scheduler = BackgroundScheduler()

def start_scheduler(configs):

    scheduler.add_job(
        call_train_api,
        trigger=CronTrigger(minute="*/1"),
        kwargs={"configs": configs}
    )

    scheduler.start()

    print("Scheduler started")

    # чтобы процесс не завершился
    import time
    while True:
        time.sleep(60)