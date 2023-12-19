from fastapi import APIRouter

from app.api.api_v1.endpoints import talents

api_router = APIRouter()
api_router.include_router(talents.router, prefix="/talents", tags=["talents"])
