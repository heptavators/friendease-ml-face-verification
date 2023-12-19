import base64
import aiohttp
import httpx
import unittest

from fastapi.testclient import TestClient
from app.main import app

LOCALHOST = "http://localhost:6969"
ID_CARD = "https://media.suara.com/suara-partners/bestie/thumbs/653x367/2023/04/05/1-foto-ktp-mimi-peri-beredar-di-media-sosial-1067209811.jpg"
SELFIE = "https://akcdn.detik.net.id/visual/2019/12/27/16be5327-8854-49ba-91dc-f3d524e29562_43.jpeg?w=480&q=90"
PROFILE_IMAGE = "https://cdn1-production-images-kly.akamaized.net/IAAtfskdccc2OCndZ-dpo1QpyAU=/1200x900/smart/filters:quality(75):strip_icc():format(webp)/kly-media-production/medias/2775841/original/045560200_1554951216-37385917_232585854247549_6306660029609017344_n.jpg"
PROFILE_IMAGE_CROWDED = "https://img.okezone.com/content/2019/08/29/194/2098210/5-bentuk-kasih-sayang-mimi-peri-kepada-emak-ratu-tak-cuma-beli-mobil-baru-XT7ODQMfFY.jpg"
OTHER_IMAGE = "https://cdn.linkumkm.id/library/9/9/4/6/4/99464_840x576.jpg"


class TestAPI(unittest.TestCase):
    client = TestClient(app)

    def __fetch_base64__(self, url: str) -> str:
        with httpx.Client() as client:
            image_binary = client.get(url).content
            return base64.b64encode(image_binary).decode("utf-8")

    def test_app_root(self):
        data = self.__fetch_json__(f"{LOCALHOST}")
        self.assertEqual(data, {"message": "Face verification API"})

    def test_verify_id_card_verified(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(SELFIE)

        payload = {"id_card": id_card, "selfie": selfie}

        response = self.client.post("api/v1/verify/id-card", json=payload).json()

        self.assertEqual(
            response, {"verified": True, "message": "Your face is verified"}
        )

    def test_verify_id_card_unverified(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(OTHER_IMAGE)

        payload = {
            "id_card": id_card,
            "selfie": selfie,
        }

        response = self.client.post(f"api/v1/verify/id-card", json=payload).json()

        self.assertEqual(
            response,
            {
                "verified": False,
                "message": "Your face is not verified! You only can upload your own images not other's",
            },
        )

    def test_verify_id_card_unverified_crowded(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(PROFILE_IMAGE_CROWDED)

        payload = {
            "id_card": id_card,
            "selfie": selfie,
        }

        response = self.client.post("api/v1/verify/id-card", json=payload).json()

        self.assertEqual(
            response,
            {
                "verified": False,
                "message": "There's more than one people in your image, make sure there's only you in the image",
            },
        )

    def test_verify_profile_verified(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(SELFIE)
        profile_image = self.__fetch_base64__(PROFILE_IMAGE)

        payload = {
            "id_card": id_card,
            "selfie": selfie,
            "profile_image": profile_image,
        }

        response = self.client.post("api/v1/verify/profile", json=payload).json()

        self.assertEqual(
            response, {"verified": True, "message": "Your face is verified"}
        )

    def test_verify_profile_unverified(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(SELFIE)
        profile_image = self.__fetch_base64__(OTHER_IMAGE)

        payload = {
            "id_card": id_card,
            "selfie": selfie,
            "profile_image": profile_image,
        }

        response = self.client.post("api/v1/verify/profile", json=payload).json()

        self.assertEqual(
            response,
            {
                "verified": False,
                "message": "Your face is not verified! You only can upload your own images not other's",
            },
        )

    def test_verify_profile_unverified_crowded(self):
        id_card = self.__fetch_base64__(ID_CARD)
        selfie = self.__fetch_base64__(SELFIE)
        profile_image = self.__fetch_base64__(PROFILE_IMAGE_CROWDED)

        payload = {
            "id_card": id_card,
            "selfie": selfie,
            "profile_image": profile_image,
        }

        response = self.client.post("api/v1/verify/profile", json=payload).json()

        self.assertEqual(
            response,
            {
                "verified": False,
                "message": "There's more than one people in your profile image, make sure there's only you in the image",
            },
        )
