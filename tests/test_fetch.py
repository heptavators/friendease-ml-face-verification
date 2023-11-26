import time
import asyncio
import httpx
import imageio.v2 as imageio
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from common import functions

URL = "https://wallpapercave.com/wp/wp8913058.jpg"


def fetch_image_sync(url):
    with httpx.Client() as client:
        image_binary = client.get(url, timeout=None).content
        image = imageio.imread(image_binary)

        # Remove the alpha channel if present
        if image.shape[-1] == 4:
            image = image[:, :, :3]

    return image.astype(dtype=np.uint8)


async def fetch_async(N):
    tasks = [functions.fetch_image(URL) for _ in range(N)]
    return await asyncio.gather(*tasks)


def fetch_thread(N, num_thread: int = None):
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        results = list(executor.map(fetch_image_sync, [URL] * N))
    return results


if __name__ == "__main__":
    N = 100

    # start = time.perf_counter()
    # loop = asyncio.get_event_loop()
    # results = loop.run_until_complete(fetch_async(N))
    # print(len(results))
    # end = time.perf_counter()
    # print(f"Fetching {N} requests takes {end-start} seconds")
    print(tuple(range(2)))
