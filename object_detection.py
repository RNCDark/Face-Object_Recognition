import cv2
from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolov8n.pt")

video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

if not video_capture.isOpened():
    raise RuntimeError("Error: Could not open video source.")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection
        results = model.predict(
            source=frame,  # Input frame
            conf=0.4,     # Confidence threshold
            verbose=False    # Disable verbose output
        )

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('YOLO Object Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()