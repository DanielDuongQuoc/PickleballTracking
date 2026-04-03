# 🐳 Hướng dẫn chạy PickleballTracking với Docker

## Yêu cầu

| Thành phần | Phiên bản tối thiểu |
|---|---|
| Docker Desktop | 24.x trở lên |
| RAM | 4 GB trống (YOLO cần ~2–3 GB khi inference) |
| Dung lượng disk | ~5 GB (image + model + dependencies) |

---

## Cấu trúc file Docker

```
Pickle_ball_tracking-main/
├── Dockerfile          # Định nghĩa image
├── docker-compose.yml  # Cấu hình service + volumes
├── .dockerignore       # Loại trừ file không cần build
└── best.pt             # Model YOLO (bắt buộc có trong thư mục)
```

---

## Hướng dẫn chạy

### Bước 1 — Mở Docker Desktop

Mở **Docker Desktop** từ Start Menu và đợi icon ở taskbar chuyển sang màu xanh (daemon sẵn sàng).

### Bước 2 — Build và chạy

```bash
# Vào thư mục project
cd PickleballTracking

# Build image và chạy container (lần đầu mất 5–15 phút)
docker compose up --build

# Hoặc chạy nền (background)
docker compose up --build -d
```

### Bước 3 — Truy cập ứng dụng

Mở trình duyệt: **http://localhost:5000**

---

## Các lệnh thường dùng

```bash
# Xem logs realtime
docker compose logs -f

# Dừng container
docker compose down

# Restart container
docker compose restart

# Xem container đang chạy
docker ps

# Vào bên trong container để debug
docker compose exec pickleballtracking bash

# Xóa image (build lại từ đầu)
docker compose down --rmi all
```

---

## Dữ liệu persistent

Hai thư mục được mount ra ngoài container:

| Thư mục (host) | Thư mục (container) | Nội dung |
|---|---|---|
| `./uploads/` | `/app/uploads/` | Video upload từ người dùng |
| `./outputs/` | `/app/outputs/` | Video + CSV sau khi tracking |

> ✅ Dữ liệu **không bị mất** khi restart hoặc xóa container.

---

## Ghi chú kỹ thuật

### Tại sao dùng `opencv-python-headless`?
Container Linux không có màn hình (no display). `opencv-python-headless` bỏ dependency Qt/GTK, nhỏ hơn ~300MB và không crash khi headless.

### Video encoding trong container
- Ghi frame bằng **XVID** (→ `.avi` tạm)
- Re-encode sang **H.264 MP4** bằng `imageio-ffmpeg` (có ffmpeg build sẵn) hoặc ffmpeg system
- Output cuối cùng trình duyệt phát được trực tiếp

### YOLO model
Model `best.pt` được copy vào image lúc build — không cần download khi chạy.

---

## Troubleshooting

### Lỗi "Cannot connect to Docker daemon"
→ Mở Docker Desktop và đợi nó khởi động hoàn tất.

### Container chạy nhưng app không response
→ YOLO đang load model lần đầu, đợi ~30-60 giây. Xem logs:
```bash
docker compose logs -f
```

### Lỗi "Out of memory" khi inference
→ Tăng memory limit trong `docker-compose.yml`:
```yaml
limits:
  memory: 6G
```

### Build thất bại do network
→ Thử build với proxy hoặc dùng pip mirror:
```dockerfile
RUN pip install --no-cache-dir -i https://pypi.org/simple/ ...
```
