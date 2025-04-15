from ultralytics import YOLO
import cv2
import time

# Load your trained model (make sure 'best.pt' is in the same folder or give full path)
model = YOLO("best.pt")  # Change if needed

# Start video capture from the webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Optional: Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Starting real-time detection...")
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Run YOLOv8 inference on the frame
        results = model.predict(source=frame, save=False, show=True, conf=0.5)

        # Optionally, do something with `results`
        # For example: extract labels, log cheating behavior, etc.

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Exiting detection loop...")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Runtime: {round(time.time() - start_time, 2)} seconds")
