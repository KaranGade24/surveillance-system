export default function DetectionList({ detections, onSelect }) {
    return (
      <div className="bg-gray-800 p-4 rounded-2xl shadow-lg overflow-y-auto max-h-[480px]">
        <h2 className="text-xl font-semibold mb-3 text-yellow-400">
          🚨 Recent Detections
        </h2>
        {detections.length === 0 && <p>No detections yet.</p>}
        <ul className="space-y-2">
          {detections.map((d, i) => (
            <li
              key={i}
              className="bg-gray-700 hover:bg-gray-600 cursor-pointer p-3 rounded-lg"
              onClick={() => onSelect(d)}
            >
              <div className="flex justify-between">
                <span className="font-medium text-white">
                  {d.object} ({(d.confidence * 100).toFixed(1)}%)
                </span>
                <span className="text-sm text-gray-400">
                  {d.timestamp}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  