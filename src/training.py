import tensorflow as tf
import tensorflow.python.keras.backend as K
from loguru import logger as log

from prepare_dataset import create_pairs

IMG_SHAPE = (64, 64, 3)


def tf_siamese_nn(shape, embedding=64, fine_tune=False):
    inputs = tf.keras.layers.Input(shape)
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=shape, include_top=False, weights='imagenet')

    if not fine_tune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - int(len(base_model.layers) * .10)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embedding)(x)
    return tf.keras.Model(inputs, outputs)


def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def contrastive_loss(y, predictions, margin=1):
    y = tf.cast(y, predictions.dtype)
    squared_predictions = K.square(predictions)
    squared_margin = K.square(K.maximum(margin - predictions, 0))
    loss = 1 - K.mean(y * squared_predictions + (1 - y) * squared_margin)
    return loss


(pairTrain, labelTrain) = create_pairs()
log.info("Dataset created")

img1 = tf.keras.layers.Input(shape=IMG_SHAPE)
img2 = tf.keras.layers.Input(shape=IMG_SHAPE)
featureExtractor = tf_siamese_nn(IMG_SHAPE)
featsA = featureExtractor(img1)
featsB = featureExtractor(img2)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
distance = tf.keras.layers.Lambda(euclidean_distance)([featsA, featsB])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
model = tf.keras.Model(inputs=[img1, img2], outputs=outputs)
model.summary()
model.compile(loss=contrastive_loss, optimizer=opt, metrics=["accuracy"], )

history = model.fit([pairTrain[:, 0], pairTrain[:, 1]],
                    labelTrain[:],
                    validation_split=0.2,
                    batch_size=8,
                    epochs=1)
model.save("face_recognition_model")
