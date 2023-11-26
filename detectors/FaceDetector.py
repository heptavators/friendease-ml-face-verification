import math
import numpy as np
import asyncio

from PIL import Image
from mtcnn import MTCNN
from common import distance
from detectors import MtcnnWrapper


def build_model() -> MTCNN:
    global face_detector_obj  # singleton design pattern

    if not "face_detector_obj" in globals():
        face_detector_obj = MtcnnWrapper.build_model()

    return face_detector_obj


def detect_faces(face_detector: MTCNN, img: np.ndarray) -> list:
    """
    This functions detects all faces in an image and return them in list of faces

    Returns:
        List of detected faces
    """
    # obj stores list of (detected_face, region, confidence)
    return MtcnnWrapper.detect_face(face_detector, img)


def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a, b, c = (
        distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd)),
        distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd)),
        distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye)),
    )

    # -----------------------

    # apply cosine rule

    if (
        b != 0 and c != 0
    ):  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway
