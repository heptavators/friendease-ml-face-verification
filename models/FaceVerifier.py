import cv2
import numpy as np

from . import Facenet
from logs import logger
from common import functions, distance as dst
from tensorflow.keras.models import Model


def build_model() -> Model:
    """
    Returns:
            built Face Verification model
    """

    # singleton design pattern
    global model_obj

    if not "model_obj" in globals():
        model_obj = Facenet.loadModel()

    return model_obj


def verify_profile(
    id_card: np.ndarray,
    selfie: np.ndarray,
    profile_image: np.ndarray,
    distance_metric: str = "cosine",
) -> dict:
    """
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            id_card, selfie, profile_image: numpy array (BGR).

            distance_metric (string): cosine, euclidean

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    "message": "Some messages"
            }

    """

    # --------------------------------
    target_size = functions.find_target_size()

    id_card_objs, selfie_objs, profile_image_objs = (
        functions.extract_faces(img=id_card, target_size=target_size),
        functions.extract_faces(img=selfie, target_size=target_size),
        functions.extract_faces(img=profile_image, target_size=target_size),
    )

    if len(profile_image_objs) > 1:
        return {
            "verified": False,
            "message": "There's more than one people in your profile image, make sure there's only you in the image",
        }

        # --------------------------------

    id_card_embedding_obj, selfie_embedding_obj, profile_image_embedding_obj = (
        represent(id_card_objs[0][0]),
        represent(selfie_objs[0][0]),
        represent(profile_image_objs[0][0]),
    )

    id_card_representation = id_card_embedding_obj[0]["embedding"]
    selfie_representation = selfie_embedding_obj[0]["embedding"]
    profile_image_representation = profile_image_embedding_obj[0]["embedding"]

    if distance_metric == "cosine":
        id_card_with_profile_distance, selfie_with_profile_distance = (
            dst.findCosineDistance(
                id_card_representation, profile_image_representation
            ),
            dst.findCosineDistance(selfie_representation, profile_image_representation),
        )
    elif distance_metric == "euclidean":
        id_card_with_profile_distance, selfie_with_profile_distance = (
            dst.findEuclideanDistance(
                id_card_representation, profile_image_representation
            ),
            dst.findEuclideanDistance(
                selfie_representation, profile_image_representation
            ),
        )
    else:
        logger.error(f"Invalid distance_metric passed - {distance_metric}")

    # -------------------------------
    threshold = dst.findThreshold(distance_metric)

    verified = (
        True
        if (
            id_card_with_profile_distance <= threshold
            or selfie_with_profile_distance <= threshold
        )
        else False
    )
    message = (
        "Your face is verified"
        if verified
        else "Your face is not verified! You only can upload your own images not other's"
    )

    return {"verified": verified, "message": message}


def verify_id_card(
    id_card: np.ndarray,
    selfie: np.ndarray,
    distance_metric: str = "cosine",
) -> dict:
    """
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            id_card, selfie: numpy array (BGR).

            distance_metric (string): cosine, euclidean

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    "message": "Some messages"
            }

    """

    # --------------------------------
    target_size = functions.find_target_size()

    id_card_objs, selfie_objs = (
        functions.extract_faces(img=id_card, target_size=target_size),
        functions.extract_faces(img=selfie, target_size=target_size),
    )

    if len(selfie_objs) > 1:
        return {
            "verified": False,
            "message": "There's more than one people in your image, make sure there's only you in the image",
        }

        # --------------------------------

    id_card_embedding_obj, selfie_embedding_obj = (
        represent(id_card_objs[0][0]),
        represent(selfie_objs[0][0]),
    )

    id_card_representation = id_card_embedding_obj[0]["embedding"]
    selfie_representation = selfie_embedding_obj[0]["embedding"]

    if distance_metric == "cosine":
        distance = dst.findCosineDistance(id_card_representation, selfie_representation)
    elif distance_metric == "euclidean":
        distance = dst.findEuclideanDistance(
            id_card_representation, selfie_representation
        )
    else:
        logger.error(f"Invalid distance_metric passed - {distance_metric}")

    # -------------------------------
    threshold = dst.findThreshold(distance_metric)

    verified = distance <= threshold
    message = (
        "Your face is verified"
        if verified
        else "Your face is not verified! You only can upload your own images not other's"
    )

    return {"verified": verified, "message": message}


def represent(
    img_arr: np.ndarray,
    normalization: str = "Facenet",
) -> list:
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_arr: numpy array (BGR)

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

    if len(img_arr.shape) == 4:
        img_arr = img_arr[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
    if len(img_arr.shape) == 3:
        img_arr = cv2.resize(img_arr, target_size)
        img_arr = np.expand_dims(img_arr, axis=0)
        # when represent is called from verify, this is already normalized
        if img_arr.max() > 1:
            img_arr /= 255
    # --------------------------------
    img_objs = [(img_arr, 0, 0)]
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
