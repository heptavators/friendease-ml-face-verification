import numpy as np
import os

from fastapi import FastAPI
from app.api.api_v1.endpoints import verify

os.environ["$WEB_CONCURRENCY"] = str(os.cpu_count() + 4)

app = FastAPI(
    title="Face Verification",
    description="API Face Verification for FriendEase Application",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"message": "Face verification API"}


app.include_router(verify.router)
