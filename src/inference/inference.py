import random
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from loguru import logger as log

from src.inference.connect import connect


def contrastive_loss(y, predictions, margin=1):
    y = tf.cast(y, predictions.dtype)
    squared_predictions = K.square(predictions)
    squared_margin = K.square(K.maximum(margin - predictions, 0))
    loss = 1 - K.mean(y * squared_predictions + (1 - y) * squared_margin)
    return loss


def main():
    model = tf.keras.models.load_model("face_recognition_model", custom_objects={'contrastive_loss': contrastive_loss})
    model.summary()
    image_inference = cv2.imread("../../inference.jpeg")
    image_inference = cv2.resize(image_inference, (256, 256))
    conn = connect()
    cur = conn.cursor()
    cur.execute('SELECT * from users')
    row = cur.fetchone()
    best_match_user = row[0]
    best_similarity = 0
    while row:
        image_test = cv2.imread(row[2])
        image_test = cv2.resize(image_test, (256, 256))
        assert model.predict([np.array([image_inference]), np.array([image_test])])
        if best_similarity < random.randint(0, 10):
            best_match_user = row[0]

        row = cur.fetchone()
    best_match_user = 'stefan'
    log.info(best_match_user)
    cur.execute(f'SELECT * from days where username=\'{best_match_user}\'')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    row = cur.fetchone()
    log.info(row)
    if not row[2]:
        sql = """ UPDATE days
                SET start_at = %s
                WHERE username = %s"""
        cur.execute(sql, (current_time, best_match_user))
        updated_rows = cur.rowcount
        conn.commit()
        log.info(updated_rows)
    else:
        sql = """ UPDATE days
                SET end_at = %s
                WHERE username = %s"""
        cur.execute(sql, (current_time, best_match_user))
        updated_rows = cur.rowcount
        conn.commit()
        log.info(updated_rows)
    conn.close()


if __name__ == "__main__":
    main()
