import asyncio
from unittest import result
import numpy as np
import time

from fastapi import FastAPI, Path, Query, Request
from pydantic import BaseModel, Field
from models import FaceVerifier


app = FastAPI(
    title="Face Verification",
    description="API Face Verification for FriendEase Application",
    version="1.0.0",
)


class FaceResult(BaseModel):
    """Result for Face Verification whether it's verified or not"""

    verified: bool = Field(description="Face is verified or not")
    distance: float = Field(
        description="The distance between template and observed image"
    )
    threshold: float = Field(description="Threshold for face verification to be passed")
    time: float = Field(description="How long does it take to verify")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/verify")
async def verify_face(payload: Request) -> dict:
    payload = await payload.json()
    template1 = payload.get("template1")
    template2 = payload.get("template2")
    profile_image = payload.get("profile_image")

    start = time.perf_counter()
    result1, result2 = await asyncio.gather(
        FaceVerifier.verify(template1, profile_image),
        FaceVerifier.verify(template2, profile_image),
    )
    end = time.perf_counter()
    print(f"Verifying needs {end-start} seconds")

    verified = False
    if result1["verified"] or result2["verified"]:
        verified = True

    return {"verified": verified}
