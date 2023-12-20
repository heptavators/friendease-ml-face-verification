from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.api.api_v1.api import api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.get("/")
def root():
    image_link = (
        "https://i.pinimg.com/564x/ac/86/3f/ac863ff709559d6d180e7a9287f2c3a4.jpg"
    )

    return HTMLResponse(
        content=f'<div style="text-align:center"><img src="{image_link}" alt="Image"></div>',
        status_code=200,
    )


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(api_router, prefix=settings.API_V1_STR)
