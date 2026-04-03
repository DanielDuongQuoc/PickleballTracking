"""
ball_in_out.py – Entry point của hệ thống Pickleball Tracking.

Cách dùng:
  python ball_in_out.py

Khi video bật lên, click chuột trái vào 4 góc sân (theo thứ tự bất kỳ).
Sau khi đủ 4 điểm, hệ thống tự động bắt đầu tracking.
Nhấn 'q' để thoát.
"""

from pathlib import Path

import cv2

from event_handler import click_on_image
from ball_tracking import BallTracker

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
video_path = BASE_DIR / '1_1.mp4'
model_path = BASE_DIR / 'best.pt'

# ── Mở video ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"[ERROR] Không mở được video: {video_path}")
    raise SystemExit(1)

# ── Đọc frame đầu tiên để setup court ────────────────────────────────────────
ret, frame = cap.read()
if not ret:
    print("[ERROR] Không đọc được frame từ video.")
    cap.release()
    raise SystemExit(1)

print("[INFO] Click trái chuột vào 4 góc sân. Nhấn 'q' để bỏ qua.")
points = click_on_image(frame)

if len(points) < 3:
    print("[WARNING] Cần ít nhất 3 điểm để tạo polygon sân. Hệ thống vẫn chạy nhưng bỏ qua kiểm tra IN/OUT.")

print(f"[INFO] Điểm sân đã chọn: {points}")

# ── Giải phóng cap trước khi BallTracker mở lại từ đầu ──────────────────────
cap.release()

# ── Bắt đầu tracking ─────────────────────────────────────────────────────────
tracker = BallTracker(str(model_path), str(video_path), points)
tracker.track()
