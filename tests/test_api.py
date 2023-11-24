import asyncio
import time
import aiohttp
import unittest
import base64

LOCALHOST = "http://127.0.0.1:8000"


class TestAPI(unittest.IsolatedAsyncioTestCase):
    async def __fetch_json__(self, endpoint: str = "") -> dict:
        URL = f"{LOCALHOST}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(URL) as response:
                return await response.json()

    async def __fetch_base64__(self, endpoint: str = "") -> dict:
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
            "template1": "https://storage.googleapis.com/payroll_anggi/test/input.png",
            "template2": "https://storage.googleapis.com/payroll_anggi/test/ktp.png",
            "profile_image": "https://storage.googleapis.com/payroll_anggi/test/output.png",
        }

        response = await self.__post__("verify", payload)

        self.assertEqual(response, {"verified": True, "alone": True})

    async def test_verify_endpoint_payload_base64_verified(self):
        template1, template2, profile_image = await asyncio.gather(
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/input.png"
            ),
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/ktp.png"
            ),
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/output.png"
            ),
        )

        payload = {
            "template1": template1,
            "template2": template2,
            "profile_image": profile_image,
        }

        response = await self.__post__("verify", payload)

        self.assertEqual(response, {"verified": True, "alone": True})

    async def test_verify_endpoint_not_verified(self):
        pass

    async def test_verify_endpoint_payload_url_multiple_requests(self):
        payload = {
            "template1": "https://storage.googleapis.com/payroll_anggi/test/input.png",
            "template2": "https://storage.googleapis.com/payroll_anggi/test/ktp.png",
            "profile_image": "https://storage.googleapis.com/payroll_anggi/test/output.png",
        }

        N = 20
        start = time.perf_counter()
        response = await asyncio.gather(
            *[self.__post__("verify", payload) for _ in range(N)]
        )
        end = time.perf_counter()
        print(f"HTTP Post {N} requests takes {end-start} seconds")

        self.assertEqual(len(response), N)

    async def test_verify_endpoint_payload_base64_multiple_requests(self):
        template1, template2, profile_image = await asyncio.gather(
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/input.png"
            ),
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/ktp.png"
            ),
            self.__fetch_base64__(
                "https://storage.googleapis.com/payroll_anggi/test/output.png"
            ),
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
