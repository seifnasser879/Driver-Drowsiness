import cv2


def open_camera(camera_index=0):

    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        return None

    return cap


def read_frame(cap):

    ret, frame = cap.read()

    if not ret:

        return None

    return frame


def release_camera(cap):

    if cap is not None:

        cap.release()