#(yolov8_env) E:\ALL_PROGRAMS\PYTHON\project#\New folder\New folder>


print("starting importing libraries..")
from ultralytics import YOLO
import cv2
print("completed")
# Step 1: Load the YOLO model
print("Loading model...")
model = YOLO("fire_detector.pt")  # Your trained YOLOv8 model
print("Model loaded successfully!")

# Step 2: Open the laptop camera (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Starting live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Step 3: Run YOLO detection on the frame
    results = model.predict(frame, verbose=False)

    # Step 4: Visualize results on the frame
    for r in results:
        annotated_frame = r.plot()  # Returns frame with boxes drawn

    # Display the frame
    cv2.imshow("YOLO Fire Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed, program ended.")
