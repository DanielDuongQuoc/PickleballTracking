"""
event_handler.py – Tiện ích click chuột để chọn 4 góc sân pickleball.
"""

import cv2


def click_on_image(image):
    """Hiển thị frame/ảnh và nhận click chuột để xác định góc sân.

    Args:
        image: numpy array (BGR) – frame video hoặc ảnh tĩnh.

    Returns:
        List[(x, y)] – danh sách tọa độ các điểm đã click (tối đa 4).
    """
    points = []
    done   = False

    # ── Handler sự kiện chuột ────────────────────────────────────────
    def click_event(event, x, y, flags, param):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN:
            # Vẽ điểm lên ảnh
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # Vẽ số thứ tự
            cv2.putText(image, str(len(points) + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            points.append((x, y))
            print(f'[Court] Điểm {len(points)}: ({x}, {y})')
            if len(points) >= 4:
                print('[Court] Đủ 4 điểm – đóng cửa sổ.')
                done = True

    # ── Hiển thị hướng dẫn lên ảnh ──────────────────────────────────
    guide = image.copy()
    cv2.putText(guide, 'Click 4 goc san, nhan Q de bo qua',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    WIN_NAME = 'Setup Court - Click 4 corners'
    cv2.imshow(WIN_NAME, guide)
    cv2.setMouseCallback(WIN_NAME, click_event)

    # ── Vòng lặp chờ click ───────────────────────────────────────────
    while not done:
        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print('[Court] Người dùng bỏ qua – dùng ảnh toàn màn hình làm sân.')
            break

    cv2.destroyAllWindows()
    return points
