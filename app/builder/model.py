import tensorflow as tf

from . import Facenet
from mtcnn import MTCNN
from app.core.config import settings


class ModelBuilder:
    _siamese_obj = None
    _facenet_obj = None
    _mtcnn_obj = None

    @staticmethod
    def get_facenet_instance():
        if ModelBuilder._facenet_obj is None:
            ModelBuilder._facenet_obj = Facenet.loadModel()
        return ModelBuilder._facenet_obj

    @staticmethod
    def get_siamese_instance():
        if ModelBuilder._siamese_obj is None:
            base_model = Facenet.InceptionResNetV2()
            last_layer = base_model.get_layer("Dropout")
            last_output = last_layer.output
            # Bottleneck
            x = tf.keras.layers.Dense(256, use_bias=False, name="Bottleneck")(
                last_output
            )
            x = tf.keras.layers.BatchNormalization(
                momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm"
            )(x)
            model = tf.keras.models.Model(base_model.input, x, name="Facenet_Model")

            model.load_weights(settings.CONFIG["model"]["siamese"])

            ModelBuilder._siamese_obj = model

        return ModelBuilder._siamese_obj

    @staticmethod
    def get_mtcnn_instance():
        if ModelBuilder._mtcnn_obj is None:
            ModelBuilder._mtcnn_obj = MTCNN()

        return ModelBuilder._mtcnn_obj


face_detector = ModelBuilder.get_mtcnn_instance()
facenet = ModelBuilder.get_facenet_instance()
siamese = ModelBuilder.get_siamese_instance()
