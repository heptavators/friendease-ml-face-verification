import asyncio
import base64
import time
import aiohttp
import unittest

LOCALHOST = "http://127.0.0.1:8000"


class TestAPI(unittest.IsolatedAsyncioTestCase):
    async def __fetch_json__(self, endpoint: str = "") -> dict:
        URL = f"{LOCALHOST}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(URL) as response:
                return await response.json()

    async def __fetch_base64__(self, endpoint: str = "") -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                image_binary = await response.read()
                return base64.b64encode(image_binary).decode("utf-8")

    async def __post__(self, endpoint: str, payload: dict) -> dict:
        URL = f"{LOCALHOST}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(URL, json=payload) as response:
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    print(f"Received unexpected content from {URL}: {response_text}")

    async def test_root_endpoint(self):
        data = await self.__fetch_json__()
        self.assertEqual(data, {"message": "Hello World"})

    async def test_verify_endpoint_payload_url_verified(self):
        payload = {
            "template1": "",
            "template2": "",
            "profile_image": "",
        }

        response = await self.__post__("verify", payload)

        self.assertEqual(response, {"verified": True})

    async def test_verify_endpoint_payload_base64_verified(self):
        template1, template2, profile_image = await asyncio.gather(
            self.__fetch_base64__(
                "https://media.suara.com/suara-partners/bestie/thumbs/653x367/2023/04/05/1-foto-ktp-mimi-peri-beredar-di-media-sosial-1067209811.jpg"
            ),
            self.__fetch_base64__(
                "https://akcdn.detik.net.id/visual/2019/12/27/16be5327-8854-49ba-91dc-f3d524e29562_43.jpeg?w=480&q=90"
            ),
            self.__fetch_base64__(
                "https://cdn1-production-images-kly.akamaized.net/IAAtfskdccc2OCndZ-dpo1QpyAU=/1200x900/smart/filters:quality(75):strip_icc():format(webp)/kly-media-production/medias/2775841/original/045560200_1554951216-37385917_232585854247549_6306660029609017344_n.jpg"
            ),
        )

        payload = {
            "template1": template1,
            "template2": template2,
            "profile_image": profile_image,
        }

        response = await self.__post__("verify", payload)

        self.assertEqual(
            response, {"verified": True, "message": "Your face is verified"}
        )

    async def test_verify_endpoint_payload_url_not_verified(self):
        pass

    async def test_verify_endpoint_payload_url_multiple_requests(self):
        payload = {
            "template1": "",
            "template2": "",
            "profile_image": "",
        }

        N = 50
        start = time.perf_counter()
        response = await asyncio.gather(
            *[self.__post__("verify", payload) for _ in range(N)]
        )
        end = time.perf_counter()
        print(f"HTTP Post {N} requests takes {end-start} seconds")

        self.assertEqual(len(response), N)

    async def test_verify_endpoint_payload_base64_multiple_requests(self):
        template1, template2, profile_image = await asyncio.gather(
            self.__fetch_base64__(""),
            self.__fetch_base64__(""),
            self.__fetch_base64__(""),
        )

        payload = {
            "template1": template1,
            "template2": template2,
            "profile_image": profile_image,
        }

        N = 100
        start = time.perf_counter()
        response = await asyncio.gather(
            *[self.__post__("verify", payload) for _ in range(N)]
        )
        end = time.perf_counter()
        print(f"HTTP Post {N} requests takes {end-start} seconds")

        self.assertEqual(len(response), N)
