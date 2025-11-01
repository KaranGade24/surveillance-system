export default function VideoPlayer({ detection, backendUrl, onClose }) {
    if (!detection) return null;
  
    const videoUrl = `${backendUrl}/play_video?timestamp=${detection.epoch_time}`;
  
    return (
      <div className="fixed inset-0 bg-black bg-opacity-70 flex justify-center items-center z-50">
        <div className="bg-gray-900 rounded-xl p-4 w-11/12 md:w-2/3">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-lg font-semibold text-blue-400">
              🎬 Detection Playback — {detection.object}
            </h2>
            <button
              onClick={onClose}
              className="bg-red-500 hover:bg-red-600 px-3 py-1 rounded-lg"
            >
              ✖ Close
            </button>
          </div>
          <video controls className="w-full rounded-lg mt-2">
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support video playback.
          </video>
        </div>
      </div>
    );
  }
  