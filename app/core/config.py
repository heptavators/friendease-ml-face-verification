import yaml

from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    CONFIG_PATH: str
    CONFIG: Dict[str, Dict[str, str]] = {}

    @validator("CONFIG", pre=True)
    def get_config(
        cls, v: Optional[Dict[str, Dict[str, str]]], values: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        if not v:
            with open(values.get("CONFIG_PATH"), "r") as config_file:
                return yaml.safe_load(config_file)
        return v

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
