import os
import cv2
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.path as mplPath


class BallTracker:
    def __init__(self, model_path, video_path, points):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.points = points
        self.kf = self.initialize_kalman_filter()

    @staticmethod
    def initialize_kalman_filter():
        """
        Khởi tạo Kalman Filter 6 chiều:
          state  = [x, y, vx, vy, ax, ay]
          sensor = [x, y]
        """
        dt = 1.0
        kf = KalmanFilter(dim_x=6, dim_z=2)

        # Trạng thái ban đầu
        kf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Covariance ban đầu (độ không chắc chắn lớn)
        kf.P = np.eye(6) * 1000.0

        # Ma trận chuyển trạng thái: x(t+1) = F * x(t)
        kf.F = np.array([
            [1, 0, dt, 0,  0.5 * dt**2, 0           ],
            [0, 1, 0,  dt, 0,            0.5 * dt**2 ],
            [0, 0, 1,  0,  dt,           0           ],
            [0, 0, 0,  1,  0,            dt          ],
            [0, 0, 0,  0,  1,            0           ],
            [0, 0, 0,  0,  0,            1           ],
        ])

        # Ma trận quan sát: z = H * x
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])

        # Nhiễu đo lường (camera noise nhỏ)
        kf.R = np.diag([5.0, 5.0])

        # Nhiễu quá trình
        q = 0.1
        kf.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0,       dt**2/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2, 0,       dt**2/2],
            [dt**3/2, 0,       dt**2,   0,       dt,      0      ],
            [0,       dt**3/2, 0,       dt**2,   0,       dt     ],
            [dt**2/2, 0,       dt,      0,       1,       0      ],
            [0,       dt**2/2, 0,       dt,      0,       1      ],
        ])
        return kf

    def track(self, progress_callback=None, output_path='output.mp4', headless=False):
        frame_num = 0
        predicted_points = []   # trail bóng (tối đa 10 điểm)
        bounce_detected = False
        last_bounce_frame = -30
        current_in_out = ""    # "IN" hoặc "OUT" – cập nhật mỗi frame theo vị trí bóng
        in_out_label = ""       # nhãn IN/OUT tại thời điểm bounce (để log)

        # FIX #1: khởi tạo next_point trước vòng lặp để tránh NameError
        next_point = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        detected = False

        # Accumulate rows, build DataFrame once at the end
        rows = []

        # ── VideoWriter ──────────────────────────────────────────────
        # Ghi tạm bằng XVID (luôn hoạt động trên Windows)
        # Sau đó re-encode sang H.264 MP4 để trình duyệt phát được
        tmp_avi = output_path.replace('.mp4', '_tmp.avi')
        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out = cv2.VideoWriter(tmp_avi, fourcc, fps, (frame_width, frame_height))

        # ── Vẽ polygon sân ngay từ đầu ──────────────────────────────
        poly_path = mplPath.Path(self.points)

        # ── Vòng lặp chính ──────────────────────────────────────────
        while True:
            ret, frame = self.cap.read()
            # FIX #2: Không có cap.grab()/retrieve() xen vào – đọc thẳng từng frame
            if not ret:
                break

            frame_num += 1
            if progress_callback and frame_num % 10 == 0:
                progress_callback(frame_num, total_frames)

            # ── YOLO detect ─────────────────────────────────────────
            bbox = self.model(frame, show=False, verbose=False)

            for boxes_1 in bbox:
                result = boxes_1.boxes.xyxy  # [N, 4] tensor

                if len(result) == 0 and not detected:
                    print(f"[Frame {frame_num}] Chưa phát hiện bóng")
                    # FIX #3: vẫn chạy Kalman predict để duy trì quỹ đạo
                    self.kf.predict()
                else:
                    detected = True

                    if len(result) != 0:
                        # Lấy box đầu tiên (chỉ track 1 bóng)
                        cx = int((result[0][0] + result[0][2]) / 2)
                        cy = int((result[0][1] + result[0][3]) / 2)
                        centroid = np.array([cx, cy], dtype=float)
                        self.kf.predict()
                        self.kf.update(centroid)
                    else:
                        # FIX #1: next_point đã được định nghĩa → dùng KF predict thay vì crash
                        cx = int(next_point[0])
                        cy = int(next_point[1])
                        centroid = np.array([cx, cy], dtype=float)
                        self.kf.predict()
                        self.kf.update(centroid)

                    prev_vy = next_point[3]        # lưu vy trước khi cập nhật
                    next_point = self.kf.x.tolist()
                    curr_vy   = next_point[3]

                    # ── Trail bóng ──────────────────────────────────
                    predicted_points.append((int(next_point[0]), int(next_point[1])))
                    if len(predicted_points) > 10:
                        predicted_points.pop(0)

                    # ── Ghi log ─────────────────────────────────────
                    rows.append({
                        'frame': frame_num,
                        'x':  next_point[0], 'y':  next_point[1],
                        'vx': next_point[2], 'vy': next_point[3],
                        'ax': next_point[4], 'ay': next_point[5],
                        'V':  np.sqrt(next_point[2]**2 + next_point[3]**2),
                    })

                    # ── Bounce Detection ─────────────────────────────
                    # Điều kiện: vy đổi từ dương (xuống) sang âm (lên)
                    # trong hệ tọa độ ảnh, y tăng = xuống
                    MIN_VY_CHANGE = 2.0   # ngưỡng để lọc noise
                    if (
                        not bounce_detected
                        and frame_num - last_bounce_frame > 25
                        and prev_vy > MIN_VY_CHANGE       # đang đi xuống
                        and curr_vy < -MIN_VY_CHANGE      # vừa đổi chiều lên
                    ):
                        bounce_detected = True
                        last_bounce_frame = frame_num
                        print(f"[Frame {frame_num}] Bounce detected!")

                        # Ghi log IN/OUT tại điểm bounce
                        ball_centroid = (next_point[0], next_point[1])
                        if poly_path.contains_point(ball_centroid):
                            in_out_label = "IN"
                            print("The ball is IN.")
                        else:
                            in_out_label = "OUT"
                            print("The ball is OUT.")

                    # Reset bounce_detected khi vy đổi chiều trở lại
                    if bounce_detected and curr_vy > 0:
                        bounce_detected = False

                    # ── Kiểm tra IN/OUT LIÊN TỤC theo vị trí bóng ──
                    ball_centroid = (next_point[0], next_point[1])
                    if poly_path.contains_point(ball_centroid):
                        current_in_out = "IN"
                    else:
                        current_in_out = "OUT"

                    # ── Vẽ ─────────────────────────────────────────
                    self._draw(frame, cx, cy, next_point,
                               predicted_points, frame_num,
                               bounce_detected, current_in_out)

            # ── Luôn ghi frame dù có detect hay không ──────────────
            out.write(frame)
            if not headless:
                cv2.imshow('PickleballTracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Cleanup
        self.cap.release()
        out.release()
        if not headless:
            cv2.destroyAllWindows()

        # ── Re-encode AVI → H.264 MP4 (browser-compatible) ──────────
        import subprocess
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            import shutil
            ffmpeg_exe = shutil.which('ffmpeg')

        if ffmpeg_exe and os.path.exists(tmp_avi):
            subprocess.run([
                ffmpeg_exe, '-y',
                '-i', tmp_avi,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',
                '-an',           # bỏ audio (video không có âm thanh)
                output_path
            ], capture_output=True)
            os.remove(tmp_avi)
            print(f'Re-encoded to H.264: {output_path}')
        else:
            # Fallback: đổi tên .avi thành .mp4 (trình duyệt có thể không phát được)
            import shutil
            if os.path.exists(tmp_avi):
                shutil.move(tmp_avi, output_path)
            print(f'Warning: ffmpeg not found, serving raw AVI as .mp4')

        # Build DataFrame once at the end
        from pathlib import Path as _P
        csv_path = str(_P(output_path).with_suffix('.csv'))
        if rows:
            test_df = pd.DataFrame(rows)
            test_df.to_csv(csv_path, index=False)
            print(f"Saved {len(rows)} rows to {csv_path}")
        return csv_path if rows else None

    # ── Helper vẽ overlay ────────────────────────────────────────────
    def _draw(self, frame, cx, cy, next_point,
              predicted_points, frame_num,
              bounce_detected, current_in_out):
        """Vẽ tất cả overlay lên frame."""
        h, w = frame.shape[:2]

        # Vẽ vùng sân (polygon)
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # Trail bóng (màu gradient cam → vàng)
        n = len(predicted_points)
        for i, p in enumerate(predicted_points):
            alpha = int(255 * (i + 1) / n)
            color = (0, alpha, 255)
            radius = max(2, int(5 * (i + 1) / n))
            cv2.circle(frame, p, radius, color, -1)

        # Vị trí detect (đỏ) và Kalman predict (xanh dương)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(frame, (int(next_point[0]), int(next_point[1])), 8, (255, 100, 0), 2)

        # HUD – thông tin text
        cv2.putText(frame, f'Frame: {frame_num}',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f'VY: {next_point[3]:.1f}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Bounce
        if bounce_detected:
            cv2.putText(frame, 'BOUNCE!',
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # ── IN / OUT liên tục – góc trên phải, to và rõ ──────────────
        if current_in_out:
            is_in      = (current_in_out == "IN")
            txt_color  = (0, 220, 0)   if is_in else (0, 0, 255)   # xanh lá / đỏ
            bg_color   = (0, 50, 0)    if is_in else (0, 0, 60)    # nền tối tương ứng
            bdr_color  = (0, 255, 60)  if is_in else (60, 60, 255) # viền sáng

            font       = cv2.FONT_HERSHEY_DUPLEX
            scale      = 3.0
            thick      = 6
            (tw, th), base = cv2.getTextSize(current_in_out, font, scale, thick)

            pad = 20
            x1  = w - tw - pad * 2 - 10
            y1  = 10
            x2  = w - 10
            y2  = th + pad * 2 + 10

            # Nền đặc
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
            # Viền nổi bật
            cv2.rectangle(frame, (x1, y1), (x2, y2), bdr_color, 4)
            # Chữ IN / OUT
            cv2.putText(frame, current_in_out,
                        (x1 + pad, y2 - pad - base),
                        font, scale, txt_color, thick, cv2.LINE_AA)
