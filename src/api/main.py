import asyncio
import cv2
import numpy as np
import time
import aiohttp
import imageio.v2 as imageio
import base64
import requests


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fastapi import FastAPI, Path, Query, Request
from pydantic import BaseModel, Field
from deepface import DeepFace


model = 'Facenet'
detector_backend = 'mtcnn'


DeepFace.build_model('Facenet')


app = FastAPI(
    title='Face Verification',
    description='API Face Verification for FriendEase Application',
    version='1.0.0'
)


class FaceResult(BaseModel):
    """ Result for Face Verification whether it's verified or not """
    
    verified: bool = Field(description="Face is verified or not")
    distance: float = Field(description="The distance between template and observed image")
    threshold: float = Field(description="Threshold for face verification to be passed")
    time: float = Field(description="How long does it take to verify")


async def verify_user_async(template, image):
    global model, detector_backend
    
    return await asyncio.to_thread(DeepFace.verify, img1_path=template, img2_path=image, model_name=model, detector_backend=detector_backend, normalization=model)


def verify_user_sync(template, image):
    global model, detector_backend
    
    return DeepFace.verify(img1_path=template, img2_path=image, model_name=model, detector_backend=detector_backend, normalization=model)


async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_binary = await response.read()
            image = imageio.imread(image_binary)
            
            # Remove the alpha channel if present
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            
            return image.astype(dtype=np.uint8)
        
        
def fetch_image_sync(url):
    response = requests.get(url, stream=True).raw
    
    image_binary = response.read()
    image = imageio.imread(image_binary)
    
    # Remove the alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    return image.astype(dtype=np.uint8)


async def decode_base64_to_image(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    image = imageio.imread(decoded_image)
    
    # Remove the alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    return image.astype(dtype=np.uint8)


def decode_base64_to_image_sync(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    image = imageio.imread(decoded_image)
    
    # Remove the alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    return image.astype(dtype=np.uint8)


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/verify')
async def verify_face(data: Request) -> dict: 
    json = await data.json()
    template1_url = json.get('template1')
    template2_url = json.get('template2')
    profile_image_url = json.get('profile_image')
    
    
    # start = time.perf_counter()
    # template1, template2, profile_image = await asyncio.gather(
    #     fetch_image(template1_url),
    #     fetch_image(template2_url),
    #     fetch_image(profile_image_url),
    # )
    # end = time.perf_counter()
    # print(f'Fetching all images needs {end-start} seconds')
    
    
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_image_sync, [template1_url, template2_url, profile_image_url]))
    end = time.perf_counter()
    print(f'Fetching all images needs {end-start} seconds')
    template1, template2, profile_image = results[0], results[1], results[2]
    
    
    start = time.perf_counter()
    results = await asyncio.gather(
        verify_user_async(template1, profile_image),
        verify_user_async(template2, profile_image),
    )
    
    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(verify_user_sync, [template1, template2], [profile_image, profile_image]))
    
    end = time.perf_counter()
    print(f'Verifying needs {end-start} seconds')
    
    
    response1 = FaceResult(verified=results[0]['verified'], distance=results[0]['distance'], threshold=results[0]['threshold'], time=results[0]['time'])
    response2 = FaceResult(verified=results[1]['verified'], distance=results[1]['distance'], threshold=results[1]['threshold'], time=results[1]['time'])
    
    
    verified = False
    if response1.verified or response2.verified:
        verified = True
    

    return {'verified': verified}


@app.post('/verify-64')
async def verify_face(data: Request) -> dict: 
    json = await data.json()
    template1_base64 = json.get('template1')
    template2_base64 = json.get('template2')
    profile_image_base64 = json.get('profile_image')
    
    
    start = time.perf_counter()
    template1, template2, profile_image = await asyncio.gather(
        decode_base64_to_image(template1_base64),
        decode_base64_to_image(template2_base64),
        decode_base64_to_image(profile_image_base64),
    )
    end = time.perf_counter()
    print(f'Decoding all images needs {end-start} seconds')
    
    
    # start = time.perf_counter()
    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(decode_base64_to_image_sync, [template1_base64, template2_base64, profile_image_base64]))
    # end = time.perf_counter()
    # print(f'Decoding all images needs {end-start} seconds')
    # template1, template2, profile_image = results[0], results[1], results[2]
    
    
    
    start = time.perf_counter()
    results = await asyncio.gather(
        verify_user_async(template1, profile_image),
        verify_user_async(template2, profile_image),
    )
    
    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(verify_user_sync, [template1, template2], [profile_image, profile_image]))
    
    end = time.perf_counter()
    print(f'Verifying needs {end-start} seconds')
    
    
    response1 = FaceResult(verified=results[0]['verified'], distance=results[0]['distance'], threshold=results[0]['threshold'], time=results[0]['time'])
    response2 = FaceResult(verified=results[1]['verified'], distance=results[1]['distance'], threshold=results[1]['threshold'], time=results[1]['time'])
    
    
    verified = False
    if response1.verified or response2.verified:
        verified = True
    

    return {'verified': verified}
