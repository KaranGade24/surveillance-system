import cv2
import numpy as np
import onnxruntime as ort

# =========================
# CONFIG
# =========================
MODEL_PATH = "fire_detector.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
TOP_K = 5          # 🔥 show only top 5 detections
SKIP_FRAMES = 2

# =========================
# LOAD MODEL
# =========================
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# =========================
# PREPROCESS
# =========================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# =========================
# POSTPROCESS (FIXED + DEBUG)
# =========================
def postprocess(outputs, frame):
    h, w, _ = frame.shape

    data = outputs[0][0]   # (5, N)

    x = data[0]
    y = data[1]
    bw = data[2]
    bh = data[3]
    conf = data[4]

    detections = []

    # Collect detections
    for i in range(len(conf)):
        raw_conf = conf[i]
        detections.append((raw_conf, x[i], y[i], bw[i], bh[i]))

    # Sort by confidence
    detections.sort(key=lambda d: d[0], reverse=True)

    #print("\nTop RAW confidences:")
        #print(detections[i][0])
#    for i in range(min(5, len(detections))):

    # 🔥 Draw top detections
    for det in detections[:5]:
        raw_conf, cx, cy, width, height = det

        # 🔥 Only draw strong detections
        if raw_conf < 0.5:
            continue

        # =========================
        # 🔥 FIXED BOX CALCULATION
        # =========================
        x1 = int(cx - width / 2)
        y1 = int(cy - height / 2)
        x2 = int(cx + width / 2)
        y2 = int(cy + height / 2)

        # Scale to original frame
        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)

        # Clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # 🚨 Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # =========================
        # 🔥 DRAW DEBUG CENTER POINT
        # =========================
        cx_draw = int(cx * w / 640)
        cy_draw = int(cy * h / 640)
        cv2.circle(frame, (cx_draw, cy_draw), 5, (255, 0, 0), -1)

        # =========================
        # 🔥 DRAW BOX
        # =========================
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(frame, f"FIRE {raw_conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)

    return frame

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0

cv2.namedWindow("🔥 Fire Detection", cv2.WINDOW_NORMAL)

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % SKIP_FRAMES != 0:
        cv2.imshow("🔥 Fire Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    try:
        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        frame = postprocess(outputs, frame)

    except Exception as e:
        print("❌ Error:", e)
        break

    cv2.imshow("🔥 Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
