import numpy as np


def findCosineDistance(source_representation, test_representation) -> float:
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


async def findEuclideanDistance(source_representation, test_representation) -> float:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def findThreshold(distance_metric: str = "cosine") -> float:
    thresholds = {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80}

    threshold = thresholds.get(distance_metric, 0.4)

    return threshold
