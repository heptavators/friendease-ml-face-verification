import numpy as np
import asyncio
import time

from fastapi import FastAPI, Path, Query, Request
from pydantic import BaseModel, Field
from models import FaceVerifier
from common import functions


app = FastAPI(
    title="Face Verification",
    description="API Face Verification for FriendEase Application",
    version="1.0.0",
)


class FaceResult(BaseModel):
    """Result for Face Verification whether it's verified or not"""

    verified: bool = Field(description="Face is verified or not")
    message: str = Field(description="Message for the face verification")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/verify")
async def verify_face(payload: Request) -> FaceResult:
    payload = await payload.json()

    template1, template2, profile_image = await asyncio.gather(
        functions.load_image(payload.get("template1")),
        functions.load_image(payload.get("template2")),
        functions.load_image(payload.get("profile_image")),
    )

    start = time.perf_counter()
    result = FaceVerifier.verify(template1, template2, profile_image)
    end = time.perf_counter()
    print(f"Verifying needs {end-start} seconds")

    response = FaceResult(verified=result["verified"], message=result["message"])

    return response
