import cv2

from ultralytics import YOLO

from config import MODEL_PATH, DEVICE


class DrowsinessDetector:

    def __init__(self):

        print(f"Loading model on: {DEVICE}")

        self.model = YOLO(MODEL_PATH)

        self.model.to(DEVICE)

        print("Model loaded successfully.")


    def predict(self, frame, confidence=0.5):

        results = self.model.predict(
            source=frame,
            device=DEVICE,
            conf=confidence,
            verbose=False
        )

        return results


    def detect_and_draw(
        self,
        frame,
        confidence=0.5
    ):

        results = self.predict(
            frame,
            confidence
        )

        for result in results:

            for box in result.boxes:

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Class
                cls = int(box.cls[0])

                # Confidence
                conf = float(box.conf[0])

                # Class name
                label_name = self.model.names[cls]

                # Color
                if label_name.lower() == "drowsy":

                    color = (0, 0, 255)

                else:

                    color = (0, 255, 0)


                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )


                # Label
                label = (
                    f"{label_name.capitalize()} "
                    f"{conf:.2f}"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        return frame