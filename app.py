import cv2
import tempfile
import streamlit as st

from detector import DrowsinessDetector
from camera import open_camera
from utils import bytes_to_image, bgr_to_rgb

from config import (
    DEVICE,
    DEFAULT_CONFIDENCE,
    DEFAULT_CAMERA_INDEX
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_detector():

    return DrowsinessDetector()


detector = get_detector()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")


st.sidebar.write(
    f"Device: `{DEVICE}`"
)


confidence = st.sidebar.slider(
    "Confidence",
    min_value=0.1,
    max_value=1.0,
    value=DEFAULT_CONFIDENCE,
    step=0.05
)


camera_index = st.sidebar.selectbox(
    "Camera",
    [0, 1],
    index=DEFAULT_CAMERA_INDEX
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚗 Driver Drowsiness Detection"
)

st.write(
    "YOLO-based driver drowsiness detection system."
)


# ============================================================
# TABS
# ============================================================

image_tab, video_tab, webcam_tab = st.tabs(
    [
        "🖼️ Image",
        "🎥 Video",
        "📷 Webcam"
    ]
)


# ============================================================
# IMAGE
# ============================================================

with image_tab:

    st.header("🖼️ Image Detection")


    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="image_upload"
    )


    if uploaded_image is not None:

        image = bytes_to_image(
            uploaded_image.read()
        )


        if image is None:

            st.error(
                "Could not read image."
            )

        else:

            result = detector.detect_and_draw(
                image.copy(),
                confidence
            )


            result = bgr_to_rgb(
                result
            )


            st.image(
                result,
                caption="Detection Result",
                use_container_width=True
            )


# ============================================================
# VIDEO
# ============================================================

with video_tab:

    st.header("🎥 Video Detection")


    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )


    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )


        temp_video.write(
            uploaded_video.read()
        )

        temp_video.close()


        cap = cv2.VideoCapture(
            temp_video.name
        )


        if not cap.isOpened():

            st.error(
                "Could not open video."
            )

        else:

            st.success(
                "Video opened successfully."
            )


            frame_placeholder = st.empty()


            while True:

                ret, frame = cap.read()


                if not ret:

                    break


                frame = detector.detect_and_draw(
                    frame,
                    confidence
                )


                frame = bgr_to_rgb(
                    frame
                )


                frame_placeholder.image(
                    frame,
                    channels="RGB",
                    use_container_width=True
                )


            cap.release()


# ============================================================
# WEBCAM
# ============================================================

with webcam_tab:

    st.header("📷 Live Webcam")


    st.write(
        f"Camera: `{camera_index}`"
    )


    start = st.button(
        "▶️ Start Camera",
        key="start_webcam"
    )


    if start:

        cap = open_camera(
            camera_index
        )


        if cap is None:

            st.error(
                f"Could not open camera {camera_index}."
            )

        else:

            st.success(
                f"Camera {camera_index} opened."
            )


            frame_placeholder = st.empty()


            while True:

                ret, frame = cap.read()


                if not ret:

                    st.error(
                        "Could not read camera frame."
                    )

                    break


                # YOLO
                frame = detector.detect_and_draw(
                    frame,
                    confidence
                )


                # BGR -> RGB
                frame = bgr_to_rgb(
                    frame
                )


                # Display
                frame_placeholder.image(
                    frame,
                    channels="RGB",
                    use_container_width=True
                )


            cap.release()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Driver Drowsiness Detection | {DEVICE}"
)