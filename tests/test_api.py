import asyncio
import numpy as np
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

    async def test_verify_endpoint_verified(self):
        urls = []
        payload = {}
        response = await self.__post__("verify", payload)

        # self.assertEqual(response, {'verified': False})

    async def test_verify_endpoint_not_verified(self):
        pass
