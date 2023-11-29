from pydantic import BaseModel, Field


class FaceResult(BaseModel):
    """Result for Face Verification whether it's verified or not"""

    verified: bool = Field(description="Face is verified or not")
    message: str = Field(description="Message for the face verification")
