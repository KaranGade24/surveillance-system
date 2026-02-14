const express = require("express");
const axios = require("axios");
// Use `sharp` for image decode/resize — more reliable and faster than Jimp for this use-case.
// We'll `require` it lazily where needed so the server can still run if not installed,
// but the dependency will be added to package.json below.
const tryRequire = (name) => {
  try {
    return require(name);
  } catch (e) {
    return null;
  }
};
const sharp = tryRequire("sharp");

const app = express();
const PORT = 5001;

// Flask MJPEG feed
// const FLASK_FEED = "http://192.168.43.108:5000/video_feed";

const FLASK_FEED =
  "https://download-statements-premises-nice.trycloudflare.com/video_feed";

// Globals
let globalFrame = null;
const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;

// -----------------------------
// Fetch MJPEG frames from Flask
// -----------------------------
function fetchFlaskFrames() {
  console.log(`Connecting to Flask MJPEG feed: ${FLASK_FEED}`);

  axios
    .get(FLASK_FEED, { responseType: "stream" })
    .then((res) => {
      console.log("✅ Connected to Flask MJPEG feed.", res.status);

      let mjpegBuffer = Buffer.alloc(0);

      res.data.on("data", async (chunk) => {
        mjpegBuffer = Buffer.concat([mjpegBuffer, chunk]);

        let start = mjpegBuffer.indexOf(Buffer.from([0xff, 0xd8])); // JPEG SOI
        let end = mjpegBuffer.indexOf(Buffer.from([0xff, 0xd9])); // JPEG EOI

        while (start !== -1 && end !== -1) {
          const jpgBuffer = mjpegBuffer.slice(start, end + 2);
          mjpegBuffer = mjpegBuffer.slice(end + 2);

          try {
            // diagnostic: ensure incoming data is a Buffer and print a short header
            // console.debug(
            //   "[frame] buffer?",
            //   Buffer.isBuffer(jpgBuffer),
            //   "len=",
            //   jpgBuffer.length,
            //   "head=",
            //   jpgBuffer.slice(0, 8).toString("hex"),
            // );

            if (sharp) {
              // preferred: decode + resize with `sharp`
              const resized = await sharp(jpgBuffer)
                .resize(FRAME_WIDTH, FRAME_HEIGHT)
                .jpeg()
                .toBuffer();
              globalFrame = resized;
            } else {
              // no image library available — stream original frame as-is
              globalFrame = jpgBuffer;
              console.warn(
                "⚠️ sharp not installed — streaming original JPEG frame (no resize)",
              );
            }
          } catch (err) {
            // print full error and stream the original JPEG so the feed keeps working
            console.error("❌ Image processing error:", err);
            if (jpgBuffer && jpgBuffer.length) {
              globalFrame = jpgBuffer;
            }
          }

          start = mjpegBuffer.indexOf(Buffer.from([0xff, 0xd8]));
          end = mjpegBuffer.indexOf(Buffer.from([0xff, 0xd9]));
        }
      });

      res.data.on("end", () => console.log("❌ Flask MJPEG stream ended."));
      res.data.on("error", (err) =>
        console.error("❌ Flask stream error:", err.message),
      );
    })
    .catch((err) =>
      console.error("❌ Error connecting to Flask MJPEG:", err.message),
    );
}

// -----------------------------
// Serve MJPEG stream to clients
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
  }, 50); // ~20 FPS

  req.on("close", () => clearInterval(interval));
});

// -----------------------------
// Start Node.js backend
// -----------------------------
app.listen(PORT, () => {
  console.log(`✅ Node.js backend running at http://localhost:${PORT}/video`);
  fetchFlaskFrames();
});
