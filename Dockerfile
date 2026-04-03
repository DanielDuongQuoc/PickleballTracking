# ─────────────────────────────────────────────────────────────
# Pickleball Tracker – Dockerfile
# Build:   docker build -t pickleball-tracker .
# Run:     docker run -p 5000:5000 pickleball-tracker
# ─────────────────────────────────────────────────────────────

# Python 3.11 slim – nhỏ gọn, đủ dùng cho headless OpenCV
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
# libgl1        → OpenCV cần để đọc/ghi video (dù headless)
# libglib2.0-0  → dependency của libGL
# ffmpeg        → fallback nếu imageio-ffmpeg không build được
# libxext6, libsm6, libxrender1 → thư viện X11 tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Cài dependencies Python trước (tận dụng Docker layer cache) ──
COPY requirements.txt .

# Dùng opencv-python-headless thay cho opencv-python
# (không cần GUI, loại bỏ Qt/GTK dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v "^opencv-python" requirements.txt > requirements_docker.txt && \
    echo "opencv-python-headless>=4.8.0" >> requirements_docker.txt && \
    pip install --no-cache-dir -r requirements_docker.txt && \
    rm requirements_docker.txt

# ── Copy source code ─────────────────────────────────────────
COPY app.py .
COPY ball_tracking.py .
COPY event_handler.py .
COPY ball_in_out.py .
COPY best.pt .
COPY templates/ ./templates/

# ── Tạo thư mục persistent (sẽ được mount ra ngoài) ─────────
RUN mkdir -p uploads outputs

# ── Non-root user (bảo mật) ──────────────────────────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Expose port ──────────────────────────────────────────────
EXPOSE 5000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ── Entrypoint ───────────────────────────────────────────────
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--graceful-timeout", "60", \
     "--keep-alive", "5", \
     "--limit-request-line", "0", \
     "--limit-request-field_size", "0", \
     "app:app"]
