from pydantic import BaseModel, Field


class VerifyIdCard(BaseModel):
    """Base payload for verifying identity card"""

    id_card: str = Field("Identity card image (base64 encoded)")
    selfie: str = Field("Selfie image (base64 encoded)")


class VerifyProfile(BaseModel):
    """Base payload for verifying profile image"""

    id_card: str = Field("Identity card image (base64 encoded)")
    selfie: str = Field("Selfie image (base64 encoded)")
    profile_image: str = Field("Profile image (base64 encoded)")


class FaceResult(BaseModel):
    """Result for Face Verification whether it's verified or not"""

    verified: bool = Field(description="Face is verified or not")
    message: str = Field(description="Message for the face verification")
