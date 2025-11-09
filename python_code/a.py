from flask import Flask, Response, request, send_file, abort
import os
import mimetypes

app = Flask(__name__)

VIDEO_DIR = r"Videos"  # same directory as in Node.js


@app.route("/")
def index():
    return "Video Streaming Server is running."

@app.route("/video/<filename>")
def stream_video(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(VIDEO_DIR, safe_name)

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        abort(404, "Video not found")

    file_size = os.path.getsize(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "video/mp4"

    range_header = request.headers.get("Range", None)

    if range_header:
        # Example: "bytes=1000-"
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
    else:
        start = 0
        end = file_size - 1

    if start >= file_size or end >= file_size:
        return Response(status=416)

    length = end - start + 1

    def generate():
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(1024 * 1024)  # 1MB chunks
            while chunk:
                yield chunk
                chunk = f.read(1024 * 1024)

    rv = Response(generate(), status=206, mimetype=mime_type)
    rv.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
    rv.headers.add("Accept-Ranges", "bytes")
    rv.headers.add("Content-Length", str(length))

    return rv


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
