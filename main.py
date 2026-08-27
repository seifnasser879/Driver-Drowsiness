import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained model
try:
  model = tf.keras.models.load_model("my_model.keras")
  print("Model loaded successfully!")
except Exception as e:
  print(f"Error loading model: {e}")
  exit()

# 2. Load OpenCV's eye detection classifier using explicit path data
cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(cascade_path)

if eye_cascade.empty():
  print(f"Error: Failed to load cascade classifier from {cascade_path}")
  exit()
else:
  print("Eye cascade loaded successfully!")

# 3. Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Error: Could not open webcam.")
  exit()

print("Testing started. Press 'q' in the video window to stop.")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame.")
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  status_text = "WARNING: Drowsy / Eyes Closed!"
  color = (0, 0, 255)

  for x, y, w, h in eyes:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Crop and resize eye region to match model input (224x224)
    eye_roi = frame[y : y + h, x : x + w]
    if eye_roi.size > 0:
      eye_resized = cv2.resize(eye_roi, (224, 224))
      eye_normalized = eye_resized / 255.0
      input_data = np.expand_dims(eye_normalized, axis=0)

      # Run prediction test
      prediction = model.predict(input_data, verbose=0)
      print(f"Prediction output: {prediction}")

      if prediction[0][0] > 0.5:  # Adjust threshold if needed
        status_text = "Testing: Eyes Open "
        color = (0, 255, 0)

  cv2.putText(
      frame,
      status_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      color,
      2,
      cv2.LINE_AA,
  )
  cv2.imshow("Model Local Test", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()