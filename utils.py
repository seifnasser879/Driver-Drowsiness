import cv2
import numpy as np


def bytes_to_image(file_bytes):

    array = np.asarray(
        bytearray(file_bytes),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return image


def bgr_to_rgb(frame):

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )