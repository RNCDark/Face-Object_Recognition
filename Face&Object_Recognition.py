import cv2
import time
from ultralytics import YOLO

# Initialize pretrained detection models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
object_model = YOLO("yolo11n.pt")

if face_cascade.empty():
    raise RuntimeError("Unable to load face detector.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to access webcam.")

show_faces = True
show_objects = True
previous_time = time.perf_counter()

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture frame.")
            break

        original_frame = frame.copy()

        # Run object detection
        if show_objects:
            results = object_model.predict(
                frame,
                conf=0.4,
                verbose=False
            )
            frame = results[0].plot()

        # Draw a filled rectangle for the label background
        cv2.rectangle(
            frame, 
            (0, 0), 
            (460, 80), 
            (30, 30, 30), 
            -1
        )
        
        # Run face detection
        if show_faces:
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Estimate processing FPS
        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "F: Faces | O: Objects | Q: Quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Computer Vision - Face and Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("f"):
            show_faces = not show_faces
        elif key == ord("o"):
            show_objects = not show_objects

finally:
    camera.release()
    cv2.destroyAllWindows()
