export default function Recordings({ recordings, backendUrl, onRefresh }) {
    return (
      <div className="bg-gray-800 p-4 rounded-2xl shadow-lg mt-6">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xl font-semibold text-purple-400">
            📁 Recordings Archive
          </h2>
          <button
            onClick={onRefresh}
            className="bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded-lg"
          >
            Refresh
          </button>
        </div>
  
        {recordings.length === 0 && <p>No recordings found.</p>}
  
        {recordings.map((day) => (
          <div key={day.date} className="mb-4">
            <h3 className="text-lg font-medium text-white">{day.date}</h3>
            {day.hours.map((h) => (
              <div key={h.hour} className="ml-4 mt-2">
                <h4 className="text-gray-400">{h.hour}</h4>
                <ul className="ml-4 list-disc">
                  {h.files.map((f, idx) => (
                    <li key={idx} className="text-sm text-gray-300">
                      <a
                        href={`${backendUrl}/${f.path}`}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:text-blue-400"
                      >
                        {f.name}
                      </a>{" "}
                      ({(f.size / 1e6).toFixed(1)} MB)
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  }
  