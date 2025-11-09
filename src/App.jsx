import { useState, useEffect } from "react";
import { io } from "socket.io-client";
import LiveFeed from "./components/LiveFeed";
import DetectionList from "./components/DetectionList";
import VideoPlayer from "./components/VideoPlayer";
import Recordings from "./components/Recordings";
import axios from "axios";

// const BACKEND_URL =
//   "https://vatican-dressing-many-vegetables.trycloudflare.com"; // ⚠️ replace with actual IP or localhost

const BACKEND_URL = "http://192.168.43.90:5000"; // ⚠️ replace with actual IP or localhost


export default function App() {
  const [detections, setDetections] = useState([]);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const socket = io(BACKEND_URL);

  // Fetch existing detections
  useEffect(() => {
    axios
      .get(`${BACKEND_URL}/detections`)
      .then((res) => setDetections(res.data));
  }, []);

  // Socket.IO real-time updates
  useEffect(() => {
    socket.on("connect", () =>
      console.log("✅ Connected to backend via Socket.IO")
    );
    socket.on("new_detection", (data) => {
      console.log("🔔 New detection received:", data);
      setDetections((prev) => [data, ...prev.slice(0, 99)]);
    });
    // return () => socket.disconnect();
  }, []);

  // Fetch recordings
  const fetchRecordings = async () => {
    const res = await axios.get(`${BACKEND_URL}/recordings`);
    setRecordings(res.data);
  };

  useEffect(() => {
    fetchRecordings();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 space-y-6">
      <h1 className="text-3xl font-bold text-center text-blue-400">
        🔥 Real-Time Surveillance Dashboard
      </h1>

      <div className="grid md:grid-cols-2 gap-6">
        <LiveFeed backendUrl={BACKEND_URL} />
        <DetectionList
          detections={detections}
          onSelect={(d) => setSelectedDetection(d)}
        />
      </div>

      {selectedDetection && (
        <VideoPlayer
          detection={selectedDetection}
          backendUrl={BACKEND_URL}
          onClose={() => setSelectedDetection(null)}
        />
      )}

      <Recordings
        recordings={recordings}
        backendUrl={BACKEND_URL}
        onRefresh={fetchRecordings}
      />
    </div>
  );
}
