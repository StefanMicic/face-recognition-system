import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

IMG_SHAPE = (64, 64)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SHAPE)
    return image


def create_pairs():
    images = []
    labels = []
    for image_file in tqdm(os.listdir('data/left')[:100]):
        image = preprocess_image(f"data/left/{image_file}")
        positive_image = preprocess_image(f"data/right/{image_file}")
        negative_image_name = image_file
        while negative_image_name == image_file:
            negative_image_name = random.choice(os.listdir("data/right"))
        negative_image = preprocess_image(f"data/right/{negative_image_name}")
        images.append([image, positive_image])
        labels.append(1)
        images.append([image, negative_image])
        labels.append(0)
    return np.asarray(images), np.asarray(labels)
