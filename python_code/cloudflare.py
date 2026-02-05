import subprocess
import threading
import time
import re
import shutil
import shutil
import os

def _which(executable):
    # Try shutil.which first, then common locations
    path = shutil.which(executable)
    if path:
        return path
    # common fallback names
    candidates = [
        "/usr/local/bin/cloudflared",
        "/usr/bin/cloudflared",
        "/snap/bin/cloudflared",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def start_cloudflare(port=5000, autorest=True):
    """Start a cloudflared tunnel to the local port and print the public URL.

    This function blocks; use `start_cloudflare_background` to run in a thread.
    """
    print("🚀 Starting Cloudflare Tunnel... Please wait.")
    print(f"💡 Start your Flask app (e.g., app.run(port={port}))")

    CLOUDFLARED_PATH = _which("cloudflared")
    if not CLOUDFLARED_PATH:
        print("❌ cloudflared binary not found in PATH or usual locations.")
        return

    while True:
        try:
            process = subprocess.Popen(
                [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                print("[cloudflared]", line)
                # Some versions print the public url in different formats
                if "trycloudflare.com" in line or "tryflutter.dev" in line or "cfargotunnel" in line:
                    match = re.search(r"https?://[^\s']+", line)
                    if match:
                        url = match.group(0)
                        print(f"\n🌍 Public URL: {url}")
                        print("✅ Cloudflare Tunnel is live. Use this URL in your frontend (BACKEND_URL).")
                        break

            process.wait()
        except Exception as e:
            print(f"❌ Failed to start cloudflared: {e}")

        if not autorest:
            break
        print("⚠️ Cloudflare Tunnel stopped. Restarting in 5 s...")
        time.sleep(5)


def start_cloudflare_background(port=5000, autorest=True):
    thread = threading.Thread(target=start_cloudflare, args=(port, autorest), daemon=True)
    thread.start()
