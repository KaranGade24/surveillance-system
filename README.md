# Surveillance System

A lightweight surveillance web application combining a React frontend and a Python-based backend for real-time object/fire/weapon detection using pretrained PyTorch / YOLO models.

This repository stitches together a web UI (Vite + React) with detection services and API endpoints implemented in Python. The project includes example detectors and model files for fire/weapon detection and an example YOLOv8 model.

## Repository layout

- `src/` - React app (Vite) source. Key components:
  - `components/LiveFeed.jsx` - live camera/video streaming UI
  - `components/DetectionList.jsx` - lists detections and metadata
  - `components/Recordings.jsx` - recorded clips viewer
  - `components/VideoPlayer.jsx` - playback component
- `public/` - static assets for the frontend
- `python_code/` - Python backend and detection scripts
  - `api.py` - (example) Python API script
  - `survillance_system_api.py` - main surveillance API (exposes endpoints used by the frontend)
  - `fire.py`, `weapon.py`, `final.py`, `a.py`, `index.html` - utilities and detection harnesses
  - `cloudflare.py` - Cloudflare integration utilities (if used)
  - pretrained model weights: `fire_detector.pt`, `weapon_detector.pt`, `yolov8n.pt`
- `package.json` - frontend dependencies and scripts
- `vite.config.js`, `tailwind.config.js`, `postcss.config.js` - frontend build/config

## Features

- Real-time video/live feed display in the web UI
- Detection pipelines for fire and weapons (PyTorch/YOLO based)
- Recordings and playback UI
- Simple API layer to serve detection results and stream frames

## Quick overview / contract

- Inputs: video stream(s) from camera(s) or uploaded video files
- Outputs: detection events (JSON), annotated frames, optional saved clips
- Error modes: missing model files, incompatible PyTorch/torchvision versions, camera access failure

## Requirements

- Node.js 16+ (recommended 18+)
- npm or yarn
- Python 3.10+ (3.8+ may work) with these typical packages installed:
  - torch (match your CUDA version if using GPU)
  - torchvision
  - opencv-python
  - numpy
  - flask or fastapi (depending on which API files you use)
  - ultralytics (optional, if YOLOv8 code is used)

Note: This repository doesn't ship a pinned `requirements.txt`. See "Backend setup" for suggested commands.

## Setup and run (frontend)

1. Install frontend deps

```bash
cd /workspaces/surveillance-system
npm install
```

2. Start dev server

```bash
npm run dev
```

This will launch the Vite dev server (by default on http://localhost:5173). Open the URL in your browser to access the UI.

If you prefer a production build:

```bash
npm run build
npm run preview
```

## Backend setup (Python)

1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install typical packages (adjust versions or add a `requirements.txt` if you maintain one):

```bash
pip install --upgrade pip
pip install flask flask-cors fastapi uvicorn opencv-python-headless numpy torch torchvision
# optionally, if using YOLOv8/Ultralytics:
pip install ultralytics
```

3. Model files

The `python_code/` folder contains pretrained model binaries: `fire_detector.pt`, `weapon_detector.pt`, and `yolov8n.pt`. Ensure these files remain in place or update the configuration to point to your model locations.

4. Run the API server

Depending on which API script is the entrypoint, start the server. Examples:

```bash
# If the project uses Flask and api.py as entrypoint
cd python_code
FLASK_APP=api.py flask run --host=0.0.0.0 --port=5000

# Or if using FastAPI/uvicorn and survillance_system_api.py
uvicorn survillance_system_api:app --host 0.0.0.0 --port 8000 --reload
```

The frontend expects the API to provide endpoints for the live feed and detection results. If you changed ports, update the frontend API base URL (usually in `src` or an .env used by Vite).

## Configuration

- Ports: frontend (Vite) default 5173, backend examples 5000 (Flask) or 8000 (uvicorn)
- Model paths: check the Python scripts in `python_code/` — they typically load model files by relative path. If you move models, update the path in the scripts or set environment variables as needed.

Suggested environment variables you may encounter or add:

- `API_HOST` - host for backend
- `API_PORT` - port for backend
- `MODEL_DIR` - directory containing `.pt` files
- `USE_GPU` - true/false toggle whether to attempt CUDA

## How the pieces connect

- Frontend: requests the live video frames and detection JSON from the backend and renders them. The `LiveFeed` component handles streaming and overlays.
- Backend: loads model(s), performs inference on frames (or on periodic snapshots), and pushes results to the frontend via HTTP responses or WebSocket (depending on implementation).

## Common commands

- Install frontend deps: `npm install`
- Run frontend dev server: `npm run dev`
- Build frontend for production: `npm run build`
- Start Flask API (example): `FLASK_APP=api.py flask run`
- Start uvicorn (FastAPI example): `uvicorn survillance_system_api:app --reload`

## Using GPU

If you have CUDA available and installed, install the appropriate `torch` wheel for your CUDA version. The detection scripts will usually attempt to detect `torch.cuda.is_available()` and move models/tensors to `cuda` automatically when `USE_GPU` or similar is set.

## Troubleshooting

- Missing model files: ensure `python_code/*.pt` files exist and paths in the Python scripts are correct.
- Torch import or CUDA errors: check the installed `torch`/CUDA compatibility. Installing `torch` via the official instructions at https://pytorch.org/get-started/locally/ is recommended.
- Camera access errors: ensure the process has permission to access the camera device, or point the app to a video file/RTSP stream.
- CORS errors: if the frontend and backend run on different hosts/ports during development, enable CORS in the backend (e.g., `flask-cors` or `fastapi.middleware.cors.CORSMiddleware`).

## Where to look in the code

- Frontend entry: `src/main.jsx` and `src/App.jsx`
- Key UI components: `src/components/LiveFeed.jsx`, `DetectionList.jsx`, `Recordings.jsx`, `VideoPlayer.jsx`
- Backend: `python_code/survillance_system_api.py`, `python_code/api.py`, `python_code/fire.py`, `python_code/weapon.py`, `python_code/final.py`

Open these files to learn how the API endpoints are named and how models are loaded. The code comments and function names should guide any required configuration.

## Contributing

1. Fork the repo and create a feature branch.
2. Add tests where appropriate, and keep changes small and focused.
3. Open a PR describing the change and test steps.

Small enhancements we recommend:

- Add a `requirements.txt` or `pyproject.toml` for reproducible Python installs.
- Add typed configuration via `.env.example` and environment variable usage in the backend.
- Add unit/integration tests for backend endpoints.

## Security and privacy notes

- Be careful with camera access and recorded footage—store recordings securely and comply with privacy laws.
- If deploying publicly, secure the API with authentication and follow best practices for serving models and streams.

## License

This repository does not include an explicit license file. Add or change a `LICENSE` file to clearly set permitted usage for your project.

## Contact / Maintainer

Maintained by the repository owner. For questions about running or extending the system, open an issue on the repo.

---

If you'd like, I can also:

- generate a `requirements.txt` with best-guess pinned packages
- add a small `.env.example` for configuring ports and model paths
- add quick start scripts to `package.json` or `python_code/` to streamline running

Tell me which of those you'd like and I'll implement them next.
