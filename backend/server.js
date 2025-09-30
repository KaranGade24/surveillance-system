// const express = require("express");
// const http = require("http");
// const { Server } = require("socket.io");
// const { PythonShell } = require("python-shell");

// const app = express();
// const server = http.createServer(app);
// const io = new Server(server, { cors: { origin: "*" } });

// io.on("connection", (socket) => {
//   console.log("Client connected");

//   socket.on("video-frame", (dataUrl) => {
//     // Call Python script for YOLO detection
//     let options = {
//       mode: "text",
//       pythonOptions: ["-u"],
//       args: [dataUrl],
//     };

//     PythonShell.run("detect_frame.py", options, function (err, results) {
//       if (err) console.error(err);
//       if (results) {
//         // Send back processed frame
//         socket.emit("processed-frame", results[0]);
//       }
//     });
//   });
// });

// server.listen(5000, () => {
//   console.log("Server running on port 5000");
// });

// server.js
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const { spawn } = require("child_process"); // for calling Python

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

const PORT = 5000;

// Serve static files if needed (frontend)
app.use(express.static("public"));

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);

  // Listen for frames from frontend
  socket.on("video-frame", (dataUrl) => {
    // Spawn Python process
    // Passing the frame as a command-line argument (base64)
    const py = spawn("python3", ["detect_frame.py", dataUrl]);

    let result = "";

    // Capture Python stdout
    py.stdout.on("data", (data) => {
      result += data.toString();
    });

    // Capture Python errors
    py.stderr.on("data", (data) => {
      console.error("Python error:", data.toString());
    });

    // When Python process exits
    py.on("close", (code) => {
      if (code === 0) {
        // Send back processed frame to frontend
        socket.emit("processed-frame", result.trim());
      } else {
        console.error(`Python exited with code ${code}`);
      }
    });
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.id);
  });
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
