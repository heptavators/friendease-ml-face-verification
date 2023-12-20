from fastapi import APIRouter

from app.api.api_v1.endpoints import verify

api_router = APIRouter()
api_router.include_router(verify.router, prefix="/verify", tags=["verify"])
