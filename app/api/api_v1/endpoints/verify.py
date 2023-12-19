import time
import asyncio
from app.schemas import schemas

from logs import logger
from app.core import functions
from app.api import FaceVerifier
from fastapi import APIRouter, Request, status

router = APIRouter(
    prefix="/verify",
    tags=["Face Verification"],
)


@router.get("/")
async def root():
    return {"message": "Endpoint for face verification"}


@router.post(
    "/id-card",
    responses={
        str(status.HTTP_500_INTERNAL_SERVER_ERROR): {
            "description": "Internal server error"
        }
    },
)
async def verify_face(payload: Request) -> schemas.FaceResult:
    try:
        payload = await payload.json()
    except:
        logger.error("Something went wrong")

    start = time.perf_counter()
    id_card, selfie = await asyncio.gather(
        functions.load_image(payload.get("id_card")),
        functions.load_image(payload.get("selfie")),
    )

    result = FaceVerifier.verify_id_card(id_card, selfie)
    end = time.perf_counter()
    print(f"Verifying needs {end-start} seconds")

    response = schemas.FaceResult(
        verified=result["verified"], message=result["message"]
    )

    return response


@router.post(
    "/profile",
    responses={
        str(status.HTTP_500_INTERNAL_SERVER_ERROR): {
            "description": "Internal server error"
        }
    },
)
async def verify_face(payload: Request) -> schemas.FaceResult:
    try:
        payload = await payload.json()
    except:
        logger.error("Something went wrong")

    start = time.perf_counter()
    id_card, selfie, profile_image = await asyncio.gather(
        functions.load_image(payload.get("id_card")),
        functions.load_image(payload.get("selfie")),
        functions.load_image(payload.get("profile_image")),
    )

    result = FaceVerifier.verify_profile(id_card, selfie, profile_image)
    end = time.perf_counter()
    print(f"Verifying needs {end-start} seconds")

    response = schemas.FaceResult(
        verified=result["verified"], message=result["message"]
    )

    return response
