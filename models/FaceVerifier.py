import time
import cv2
import asyncio
import numpy as np

from . import Facenet
from logs import logger
from common import functions, distance as dst
from tensorflow.keras.models import Model


def build_model() -> Model:
    """
    This function builds a Face Verification model

    Returns:
            built Face Verification model
    """

    # singleton design pattern
    global model_obj

    if not "model_obj" in globals():
        model_obj = Facenet.loadModel()

    return model_obj


async def verify(
    template1: str,
    template2: str,
    profile_image: str,
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    normalization="Facenet",
) -> dict:
    """
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            template1, template2, profile_image: image url or based64 encoded
            If one of pair has more than one face, then we will immediately return False

            distance_metric (string): cosine, euclidean

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    "alone": True
            }

    """

    template1, template2, profile_image = await asyncio.gather(
        functions.load_image(template1),
        functions.load_image(template2),
        functions.load_image(profile_image),
    )

    # --------------------------------
    target_size = functions.find_target_size()

    # img pairs might have many faces
    template1_objs, template2_objs, profile_objs = await asyncio.gather(
        functions.extract_faces(
            img=template1,
            target_size=target_size,
            grayscale=False,
            enforce_detection=enforce_detection,
        ),
        functions.extract_faces(
            img=template2,
            target_size=target_size,
            grayscale=False,
            enforce_detection=enforce_detection,
        ),
        functions.extract_faces(
            img=profile_image,
            target_size=target_size,
            grayscale=False,
            enforce_detection=enforce_detection,
        ),
    )

    if len(profile_objs) > 1:
        return {"verified": False, "alone": False}

    # --------------------------------
    (
        template1_embedding_obj,
        template2_embedding_obj,
        profile_embedding_obj,
    ) = await asyncio.gather(
        represent(
            img=template1_objs[0][0],
            detector_backend="skip",
            normalization=normalization,
        ),
        represent(
            img=template2_objs[0][0],
            detector_backend="skip",
            normalization=normalization,
        ),
        represent(
            img=profile_objs[0][0],
            detector_backend="skip",
            normalization=normalization,
        ),
    )

    template1_representation = template1_embedding_obj[0]["embedding"]
    template2_representation = template2_embedding_obj[0]["embedding"]
    profile_representation = profile_embedding_obj[0]["embedding"]

    if distance_metric == "cosine":
        distance1, distance2 = await asyncio.gather(
            dst.findCosineDistance(template1_representation, profile_representation),
            dst.findCosineDistance(template2_representation, profile_representation),
        )
    elif distance_metric == "euclidean":
        distance1, distance2 = await asyncio.gather(
            dst.findEuclideanDistance(template1_representation, profile_representation),
            dst.findEuclideanDistance(template2_representation, profile_representation),
        )
    else:
        logger.error(f"Invalid distance_metric passed - {distance_metric}")

    threshold = dst.findThreshold(distance_metric)

    resp_obj = {
        "verified": True
        if (distance1 <= threshold or distance2 <= threshold)
        else False,
        "alone": True,
    }

    return resp_obj


async def represent(
    img: np.ndarray,
    detector_backend: str,
    normalization: str = "Facenet",
) -> dict:
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img (string): numpy array (BGR).

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object with multidimensional vector (embedding).
            The number of dimensions is changing based on the reference model.
            E.g. FaceNet returns 128 dimensional vector
    """
    resp_objs = []

    model = build_model()

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = functions.find_target_size()
    if detector_backend != "skip":
        img_objs = await functions.extract_faces(
            img=img,
            target_size=target_size,
            detector_backend=detector_backend,
        )
    else:
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            # when represent is called from verify, this is already normalized
            if img.max() > 1:
                img /= 255

            img_objs = [(img, 0, 0)]
    # ---------------------------------

    for img, _, _ in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        if "keras" in str(type(model)):
            # model.predict causes memory issue when it is called in a for loop
            # embedding = model.predict(img, verbose=0)[0].tolist()
            embedding = model(img, training=False).numpy()[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_objs.append(resp_obj)

    return resp_objs
