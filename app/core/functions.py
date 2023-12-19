import os
import base64
import numpy as np
import cv2
import aiohttp
import imageio.v3 as iio

from pathlib import Path
from keras.preprocessing import image
from app.detectors import FaceDetector
from app.core.logs import logger


def initialize_folder() -> None:
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_facenet_home()
    facenet_home_path = home + "/.facenet"
    weight_path = facenet_home_path + "/weights"

    if not os.path.exists(facenet_home_path):
        os.makedirs(facenet_home_path, exist_ok=True)
        logger.info("Directory ", home, "/.facenet created")

    if not os.path.exists(weight_path):
        os.makedirs(weight_path, exist_ok=True)
        logger.info("Directory ", home, "/.facenet/weights created")


def get_facenet_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("FACENET_HOME", default=str(Path.home())))


async def fetch_image(url) -> np.ndarray:
    """Load image from url.

    Args:
        url: image url.

    Returns:
        numpy array: the loaded image.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_binary = await response.read()
            image = iio.imread(image_binary)

            # Remove the alpha channel if present
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            return image.astype(dtype=np.uint8)


async def load_base64_image(uri: str) -> np.ndarray:
    """Load image from base64 string.

    Args:
        uri: a base64 string.

    Returns:
        numpy array: the loaded image.
    """

    decoded_image = base64.b64decode(uri)
    image = iio.imread(decoded_image)

    # Remove the alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    return image.astype(dtype=np.uint8)


async def load_image(img: str) -> np.ndarray:
    """Load image from url or base64.

    Args:
        img: an url or base64.

    Raises:
        ValueError: if the image path does not exist.

    Returns:
        numpy array: the loaded image.
    """

    # The image is a url
    if img.startswith("http"):
        return await fetch_image(img)
    # The image is a base64 string
    elif isinstance(img, str):
        return await load_base64_image(img)
    else:
        logger.error("Can't load the image, make it's either an url or base64")


def extract_faces(
    img: np.ndarray,
    target_size: tuple,
    detector_backend: str = "mtcnn",
    grayscale: bool = False,
    enforce_detection: bool = True,
) -> list:
    """Extract faces from an image.

    Args:
        img: numpy array (BGR).
        target_size (tuple): the target size of the extracted faces.
        detector_backend (str, optional): the face detector backend. Defaults to "mtcnn".
        grayscale (bool, optional): whether to convert the extracted faces to grayscale.
        Defaults to False.
        enforce_detection (bool, optional): whether to enforce face detection. Defaults to True.

    Raises:
        ValueError: if face could not be detected and enforce_detection is True.

    Returns:
        list: a list of extracted faces.
    """

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == "skip":
        face_objs = [(img, img_region, 0)]
    else:
        face_detector = FaceDetector.build_model()
        face_objs = FaceDetector.detect_faces(face_detector, img)

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        logger.error(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False."
        )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                # resize and padding
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(current_img.shape[1] * factor),
                    int(current_img.shape[0] * factor),
                )
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection == True:
        logger.error(
            f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
        )

    return extracted_faces


def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image.
    """

    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == "base":
        return img

    # @trevorgribble and @davedgd contributed this feature
    # restore input in scale of [0, 255] because it was normalized in scale of
    # [0, 1] in preprocess_face
    img *= 255

    if normalization == "raw":
        pass  # return just restored pixels

    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        logger.error(f"Unimplemented normalization type - {normalization}")

    return img


def find_target_size() -> tuple:
    """Find the target size of the model.

    Returns:
        tuple: the target size.
    """

    target_size = (160, 160)

    return target_size
