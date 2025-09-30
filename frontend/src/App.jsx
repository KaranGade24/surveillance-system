import React, { useEffect, useRef } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = "https://studious-doodle-9p4j59v5vwphqq7-5000.app.github.dev/";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // Initialize socket
    socketRef.current = io(SOCKET_URL);

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error("Error accessing webcam:", err);
      });

    // Listen for processed frames from backend
    socketRef.current.on("processed-frame", (dataUrl) => {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        const img = new Image();
        img.src = dataUrl;
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
        };
      }
    });

    return () => {
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  useEffect(() => {
    const sendFrame = () => {
      if (!videoRef.current || !canvasRef.current || !socketRef.current) return;

      const ctx = canvasRef.current.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      const dataUrl = canvasRef.current.toDataURL("image/jpeg");
      socketRef.current.emit("video-frame", dataUrl);

      requestAnimationFrame(sendFrame);
    };

    if (videoRef.current) {
      videoRef.current.addEventListener("play", sendFrame);
    }

    return () => {
      if (videoRef.current) videoRef.current.removeEventListener("play", sendFrame);
    };
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1>Live YOLO Detection</h1>
      <video
        ref={videoRef}
        autoPlay
        muted
        width={640}
        height={480}
        style={{ border: "1px solid black" }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ border: "1px solid black", display: "block", margin: "20px auto" }}
      />
    </div>
  );
};

export default App;
