
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Benchmark configuration
TEST_DURATION = 30
WARMUP_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.4

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load detection models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
object_model = YOLO("yolo11n.pt")

if face_cascade.empty():
    raise RuntimeError("Unable to load face classifier.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to access webcam.")


def detect(frame, use_faces, use_objects):
    """Run the selected detection methods on a webcam frame."""

    if use_faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

    if use_objects:
        object_model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )


def warmup():
    """Warm up the object detector before benchmarking."""

    print("Warming up YOLO...")

    for _ in range(WARMUP_FRAMES):
        success, frame = camera.read()

        if not success:
            raise RuntimeError("Unable to read webcam frame.")

        object_model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )


def run_benchmark(name, use_faces, use_objects):
    """Measure processing performance for one configuration."""

    print(f"\nRunning {name} for {TEST_DURATION} seconds...")

    frames_processed = 0
    total_processing_time = 0.0

    start_time = time.perf_counter()

    while time.perf_counter() - start_time < TEST_DURATION:
        success, frame = camera.read()

        if not success:
            print("Unable to capture frame.")
            break

        processing_start = time.perf_counter()

        detect(frame, use_faces, use_objects)

        processing_time = time.perf_counter() - processing_start

        total_processing_time += processing_time
        frames_processed += 1

    elapsed_time = time.perf_counter() - start_time

    if frames_processed == 0:
        raise RuntimeError(f"No frames processed for {name}.")

    average_fps = frames_processed / elapsed_time
    average_processing_ms = (
        total_processing_time / frames_processed
    ) * 1000

    result = {
        "mode": name,
        "duration_seconds": round(elapsed_time, 2),
        "frames_processed": frames_processed,
        "average_fps": round(average_fps, 2),
        "average_processing_ms": round(average_processing_ms, 2)
    }

    print(result)

    return result


def save_results(results):
    """Export benchmark results to CSV."""

    csv_path = RESULTS_DIR / "performance.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")


def create_chart(results):
    """Create a bar chart comparing average FPS."""

    modes = [result["mode"] for result in results]
    fps_values = [result["average_fps"] for result in results]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(modes, fps_values)

    plt.title("Face and Object Detection Performance")
    plt.xlabel("Detection Mode")
    plt.ylabel("Average FPS")

    for bar, value in zip(bars, fps_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0, max(fps_values) * 1.2)
    plt.tight_layout()

    chart_path = RESULTS_DIR / "performance.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Chart saved to: {chart_path}")


def main():
    configurations = [
        ("Face Only", True, False),
        ("Object Only", False, True),
        ("Combined", True, True)
    ]

    try:
        warmup()

        results = []

        for name, use_faces, use_objects in configurations:
            result = run_benchmark(
                name,
                use_faces,
                use_objects
            )
            results.append(result)

        save_results(results)
        create_chart(results)

        print("\nAll benchmarks completed successfully!")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
