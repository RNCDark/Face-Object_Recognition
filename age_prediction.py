
import cv2
from pathlib import Path
from collections import deque
import numpy as np

# Load the pretrained Haar Cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Failed to load the Haar Cascade classifier.")

# Load the pretrained age prediction CNN
model_dir = Path(__file__).resolve().parent / "models"

age_model = str(model_dir / "age_net.caffemodel")
age_config = str(model_dir / "age_deploy.prototxt")

age_net = cv2.dnn.readNetFromCaffe(age_config, age_model)

# Age categories predicted by the CNN
age_ranges = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

# Mean values used during model preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Initialize webcam
video_capture = cv2.VideoCapture(0)


if not video_capture.isOpened():
    raise RuntimeError("Unable to access the webcam.")

# Store the most recent 10 prediction vectors
prediction_history = deque(maxlen=10)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture webcam frame.")
        break

    # Convert frame to grayscale for face detection
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Improve local contrast for face detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80)
    )

    # Process each detected face
    for (x, y, w, h) in faces:

        # Extract the facial region from the original color frame
        padding = int(0.2 * w)  # Add padding around the face
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame_width, x + w + padding)
        y2 = min(frame_height, y + h + padding)
        face = frame[y1:y2, x1:x2]

        # Prepare the face for the CNN
        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0,
            size=(227, 227),
            mean=MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False
        )

        # Predict the age range
        age_net.setInput(blob)
        predictions = age_net.forward()

        # Store the current model prediction
        prediction_history.append(predictions.flatten())

        # Average the predictions over the last 10 frames
        smoothed_predictions = np.mean(prediction_history, axis=0)

        # Select the highest scoring age range
        age_index = np.argmax(smoothed_predictions)

        predicted_age = age_ranges[age_index]
        confidence = float(smoothed_predictions[age_index])

        # Draw face bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display predicted age and confidence
        label = f"Age: {predicted_age} ({confidence * 100:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display detected face count
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Age Prediction - OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()