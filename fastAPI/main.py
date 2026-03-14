from fastapi import FastAPI

from fastAPI.train_api import router

app = FastAPI()

app.include_router(router)
