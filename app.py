"""
app.py – Flask web server for PickleballTracking System.

Usage:
    py app.py
Then open http://localhost:5000 in your browser.
"""

import os
import re
import uuid
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, request, jsonify, send_file,
                   render_template_string, Response)

from ball_tracking import BallTracker

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / 'uploads'
OUTPUT_DIR  = BASE_DIR / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
MODEL_PATH = str(BASE_DIR / 'best.pt')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


# ── Restore completed jobs from disk on startup ───────────────────────────────
def restore_jobs_from_disk():
    """Scan outputs/ and re-register any previously completed jobs."""
    for mp4 in OUTPUT_DIR.glob('*_output.mp4'):
        job_id = mp4.stem.replace('_output', '')
        if job_id not in jobs:
            csv_path = mp4.with_suffix('.csv')
            jobs[job_id] = {
                'status':      'done',
                'video_path':  '',
                'progress':    1,
                'total':       1,
                'fps':         30,
                'width':       0,
                'height':      0,
                'bounces':     0,
                'message':     'Completed (restored from disk)',
                'output_path': str(mp4),
                'csv_path':    str(csv_path) if csv_path.exists() else None,
            }

restore_jobs_from_disk()


# ── HTML template ─────────────────────────────────────────────────────────────
HTML = open(BASE_DIR / 'templates' / 'index.html', encoding='utf-8').read()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_output_mp4(job_id: str) -> Path | None:
    """Find the output MP4 for a job_id — first from jobs dict, then disk."""
    # 1. Try jobs dict
    if job_id in jobs:
        p = jobs[job_id].get('output_path')
        if p and Path(p).exists():
            return Path(p)
    # 2. Fallback: scan outputs/ directory
    candidate = OUTPUT_DIR / f'{job_id}_output.mp4'
    if candidate.exists():
        return candidate
    return None


def _find_output_csv(job_id: str) -> Path | None:
    """Find the output CSV for a job_id."""
    if job_id in jobs:
        p = jobs[job_id].get('csv_path')
        if p and Path(p).exists():
            return Path(p)
    candidate = OUTPUT_DIR / f'{job_id}_output.csv'
    if candidate.exists():
        return candidate
    return None


def _stream_file(path: Path, content_type: str):
    """Stream a file with full HTTP Range-request support."""
    file_size = path.stat().st_size
    range_header = request.headers.get('Range', None)

    if range_header:
        byte_start, byte_end = 0, file_size - 1
        m = re.search(r'bytes=(\d*)-(\d*)', range_header)
        if m:
            s, e = m.group(1), m.group(2)
            if s:
                byte_start = int(s)
            if e:
                byte_end = int(e)
        length = byte_end - byte_start + 1

        def generate():
            with open(path, 'rb') as f:
                f.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            'Content-Range':  f'bytes {byte_start}-{byte_end}/{file_size}',
            'Accept-Ranges':  'bytes',
            'Content-Length': str(length),
            'Content-Type':   content_type,
        }
        return Response(generate(), 206, headers=headers)

    # Full file
    response = send_file(str(path), mimetype=content_type,
                         as_attachment=False, conditional=True)
    response.headers['Accept-Ranges'] = 'bytes'
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    """Accept a video file, save it, return the first frame as base64 JPEG."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    f = request.files['video']
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    job_id   = str(uuid.uuid4())
    vid_path = UPLOAD_DIR / f'{job_id}{ext}'
    f.save(str(vid_path))

    # Read first frame for court setup
    cap = cv2.VideoCapture(str(vid_path))
    ok, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ok:
        return jsonify({'error': 'Cannot read video file'}), 400

    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    frame_b64 = base64.b64encode(buf).decode()

    jobs[job_id] = {
        'status':      'ready',
        'video_path':  str(vid_path),
        'progress':    0,
        'total':       total,
        'fps':         fps,
        'width':       w,
        'height':      h,
        'bounces':     0,
        'message':     'Waiting for court setup',
        'output_path': None,
        'csv_path':    None,
    }

    return jsonify({
        'job_id':        job_id,
        'frame':         frame_b64,
        'width':         w,
        'height':        h,
        'total_frames':  total,
        'fps':           fps,
    })


@app.route('/track', methods=['POST'])
def start_tracking():
    """Start background tracking for a given job_id + court points."""
    data   = request.json or {}
    job_id = data.get('job_id')
    points = data.get('points', [])

    if not job_id or job_id not in jobs:
        return jsonify({'error': 'Invalid job_id'}), 400
    if len(points) < 3:
        return jsonify({'error': 'Need at least 3 court points'}), 400

    job = jobs[job_id]
    if job['status'] == 'running':
        return jsonify({'error': 'Job already running'}), 409

    out_path = str(OUTPUT_DIR / f'{job_id}_output.mp4')
    csv_path = str(OUTPUT_DIR / f'{job_id}_output.csv')

    job['status']      = 'running'
    job['progress']    = 0
    job['bounces']     = 0
    job['output_path'] = out_path
    job['csv_path']    = csv_path
    job['message']     = 'Starting model…'

    def run():
        try:
            tracker = BallTracker(
                model_path=MODEL_PATH,
                video_path=job['video_path'],
                points=[tuple(p) for p in points],
            )

            def on_progress(frame_num, total_frames):
                job['progress'] = frame_num
                job['total']    = total_frames
                job['message']  = f'Processing frame {frame_num}/{total_frames}'

            tracker.track(
                progress_callback=on_progress,
                output_path=out_path,
                headless=True,
            )

            # Verify output exists and has content
            output_file = Path(out_path)
            if not output_file.exists() or output_file.stat().st_size < 1000:
                job['status']  = 'error'
                job['message'] = f'Output video missing or empty ({output_file.stat().st_size if output_file.exists() else 0} bytes)'
                return

            job['status']   = 'done'
            job['progress'] = job['total']
            job['message']  = 'Completed!'
            print(f'[Job {job_id[:8]}] Done — {output_file.stat().st_size / 1024 / 1024:.1f} MB')

        except Exception as e:
            import traceback
            job['status']  = 'error'
            job['message'] = str(e)
            print(traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status': 'started', 'job_id': job_id})


@app.route('/status/<job_id>')
def job_status(job_id):
    # Fallback: reconstruct from disk if not in memory
    if job_id not in jobs:
        mp4 = _find_output_mp4(job_id)
        if mp4:
            csv = _find_output_csv(job_id)
            return jsonify({
                'status':     'done',
                'progress':   1,
                'total':      1,
                'percent':    100,
                'bounces':    0,
                'message':    'Completed (restored)',
                'has_output': True,
                'has_csv':    csv is not None,
            })
        return jsonify({'error': 'Job not found'}), 404

    j = jobs[job_id]
    pct = 0
    if j['total']:
        pct = min(100, round(j['progress'] / j['total'] * 100, 1))
    return jsonify({
        'status':     j['status'],
        'progress':   j['progress'],
        'total':      j['total'],
        'percent':    pct,
        'bounces':    j.get('bounces', 0),
        'message':    j['message'],
        'has_output': j['output_path'] is not None and Path(j['output_path']).exists(),
        'has_csv':    j['csv_path']    is not None and Path(j['csv_path']).exists(),
    })


@app.route('/download/video/<job_id>')
def download_video(job_id):
    """Stream video with HTTP Range support — browser <video> can seek."""
    path = _find_output_mp4(job_id)
    if not path:
        return jsonify({'error': 'Output video not found'}), 404
    return _stream_file(path, 'video/mp4')


@app.route('/download/csv/<job_id>')
def download_csv(job_id):
    path = _find_output_csv(job_id)
    if not path:
        return jsonify({'error': 'CSV not found'}), 404
    return send_file(str(path), mimetype='text/csv',
                     as_attachment=True,
                     download_name=f'tracking_{job_id[:8]}.csv')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  PickleballTracking Web App")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
