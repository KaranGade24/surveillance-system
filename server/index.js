const ioClient = require("socket.io-client");
const Jimp = require("jimp");
const ort = require("onnxruntime-node");
const express = require("express");
const { Readable } = require("stream");

const app = express();
const PORT = 5001;
const FLASK_SOCKET = "http://192.168.43.108:5000";

// Globals
let globalFrame = null;
let session;

// -----------------------------
// Load ONNX Fire Detection model
// -----------------------------
async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./firedetect8n.onnx");
    console.log("✅ Fire Detection ONNX model loaded!");
  } catch (err) {
    console.error("❌ Failed to load ONNX model:", err);
  }
}

// -----------------------------
// Fire detection function
// -----------------------------
async function detectFire(jimpImg) {
  if (!session) return jimpImg;

  const buffer = await jimpImg.getBufferAsync(Jimp.MIME_JPEG);
  const img = await Jimp.read(buffer);
  img.resize(640, 480); // Resize for model if needed

  // Convert image to CHW float32 array
  const tensorData = new Float32Array(3 * img.bitmap.height * img.bitmap.width);
  let idx = 0;
  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, offset) {
    tensorData[idx++] = this.bitmap.data[offset] / 255; // R
    tensorData[idx++] = this.bitmap.data[offset + 1] / 255; // G
    tensorData[idx++] = this.bitmap.data[offset + 2] / 255; // B
  });

  const tensor = new ort.Tensor("float32", tensorData, [
    1,
    3,
    img.bitmap.height,
    img.bitmap.width,
  ]);
  const feeds = { images: tensor }; // adjust input name as needed

  try {
    const results = await session.run(feeds);
    // TODO: parse results to get bounding boxes
    // For demo, just annotate "Fire Detection" text
    img.print(
      await Jimp.loadFont(Jimp.FONT_SANS_16_BLACK),
      10,
      10,
      "🔥 Fire Detection"
    );
  } catch (err) {
    console.error("Error running ONNX model:", err);
  }

  return img;
}

// -----------------------------
// Connect to Flask Socket.IO
// -----------------------------
const socket = ioClient(FLASK_SOCKET);

socket.on("connect", () => console.log("✅ Connected to Flask Socket.IO"));

socket.on("frame", async (data) => {
  try {
    const imgBuffer = Buffer.from(data.image, "base64");
    let img = await Jimp.read(imgBuffer);

    img = await detectFire(img); // run fire detection

    globalFrame = await img.getBufferAsync(Jimp.MIME_JPEG);
  } catch (err) {
    console.error("❌ Error processing frame:", err);
  }
});

// -----------------------------
// Serve MJPEG stream
// -----------------------------
app.get("/video", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=frame",
    "Cache-Control": "no-cache",
    Connection: "close",
    Pragma: "no-cache",
  });

  const interval = setInterval(() => {
    if (!globalFrame) return;
    res.write(`--frame\r\nContent-Type: image/jpeg\r\n\r\n`);
    res.write(globalFrame);
    res.write("\r\n");
  }, 50);

  req.on("close", () => clearInterval(interval));
});

// -----------------------------
// Start Node.js backend
// -----------------------------
app.listen(PORT, async () => {
  console.log(
    `Node.js MJPEG backend running at http://localhost:${PORT}/video`
  );
  //   await loadModel();
});
