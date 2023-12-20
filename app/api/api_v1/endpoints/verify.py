import time
import asyncio
from app.schemas import VerifyIdCard, VerifyProfile, FaceResult

from app.core import functions
from app.core.logs import logger
from app.api import FaceVerifier
from fastapi import APIRouter, Request, status

router = APIRouter()


@router.get("")
def root():
    return {"message": "Verify endpoint"}


@router.post(
    "/id-card",
    responses={
        str(status.HTTP_500_INTERNAL_SERVER_ERROR): {
            "description": "Internal server error"
        }
    },
)
async def verify_id_card(payload: VerifyIdCard) -> FaceResult:
    start = time.perf_counter()
    id_card, selfie = await asyncio.gather(
        functions.load_image(payload.id_card),
        functions.load_image(payload.selfie),
    )

    result = FaceVerifier.verify_id_card(id_card, selfie)
    end = time.perf_counter()
    logger.debug(f"Verifying id-card needs {end-start} seconds")

    response = FaceResult(verified=result["verified"], message=result["message"])

    return response


@router.post(
    "/profile",
    responses={
        str(status.HTTP_500_INTERNAL_SERVER_ERROR): {
            "description": "Internal server error"
        }
    },
)
async def verify_profile(payload: VerifyProfile) -> FaceResult:
    start = time.perf_counter()
    id_card, selfie, profile_image = await asyncio.gather(
        functions.load_image(payload.id_card),
        functions.load_image(payload.selfie),
        functions.load_image(payload.profile_image),
    )

    result = FaceVerifier.verify_profile(id_card, selfie, profile_image)
    end = time.perf_counter()
    logger.debug(f"Verifying profile needs {end-start} seconds")

    response = FaceResult(verified=result["verified"], message=result["message"])

    return response
