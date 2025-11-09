from ultralytics import YOLO
import cv2

print("Starting importing libraries...")
# Load models
fire_model = YOLO("fire_detector.pt")
weapon_model = YOLO("weapon_detector.pt")
print("Models loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Starting live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection for both models
    fire_results = fire_model.predict(frame, verbose=False)
    weapon_results = weapon_model.predict(frame, verbose=False)

    # Make a copy of the frame to annotate
    annotated_frame = frame.copy()

    # Draw fire detections (red)
    for fire_r in fire_results:
        for box in fire_r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"Fire {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw weapon detections (blue)
    for weapon_r in weapon_results:
        for box in weapon_r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"Weapon {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display combined frame
    cv2.imshow("Fire + Weapon Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed, program ended.")
