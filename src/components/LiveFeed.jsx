export default function LiveFeed({ backendUrl }) {
    return (
      <div className="bg-gray-800 p-4 rounded-2xl shadow-lg">
        <h2 className="text-xl font-semibold mb-3 text-green-400">
          📡 Live Detection Feed
        </h2>
        <div className="w-full aspect-video bg-black rounded-lg overflow-hidden">
          <img
            src={`${backendUrl}/video_feed`}
            alt="Live Feed"
            className="w-full h-full object-cover"
          />
        </div>
      </div>
    );
  }
  