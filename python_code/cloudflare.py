import subprocess, threading, time, re

# Path to your installed cloudflared.exe
CLOUDFLARED_PATH = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"

def start_cloudflare(port=5000, autorest=True):
    print("🚀 Starting Cloudflare Tunnel... Please wait.")
    print(f"💡 Start your Flask app (e.g., app.run(port={port}))")
    print(f"Using cloudflared at: {CLOUDFLARED_PATH}")

    while True:
        try:
            process = subprocess.Popen(
                [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in process.stdout:
                if "trycloudflare.com" in line:
                    match = re.search(r"https://[^\s]+\.trycloudflare\.com", line)
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
