import cv2

# Đường dẫn đến video
video_path = '/home/maicg/Downloads/toan_canh_hop_quoc_hoi_ngay_27_5_ky_hop_thu_5_quoc_hoi_khoa_xv_vnews.mp4'

# Mở video để đọc
video = cv2.VideoCapture(video_path)

# Đọc khung hình đầu tiên
ret, previous_frame = video.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Đọc khung hình tiếp theo
    ret, current_frame = video.read()

    # Kiểm tra xem còn khung hình nữa không
    if not ret:
        break

    # Chuyển đổi khung hình hiện tại sang ảnh grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Tính toán Optical Flow
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Vẽ các vector Optical Flow lên khung hình
    h, w = previous_gray.shape
    flow_visualize = cv2.cvtColor(previous_gray, cv2.COLOR_GRAY2BGR)

    for y in range(0, h, 10):
        for x in range(0, w, 10):
            dx, dy = flow[y, x]
            cv2.arrowedLine(flow_visualize, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), 1)

    # Hiển thị khung hình với các vector Optical Flow
    cv2.imshow('Optical Flow', flow_visualize)

    # Cập nhật khung hình trước là khung hình hiện tại
    previous_gray = current_gray

    # Kiểm tra phím nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video.release()
cv2.destroyAllWindows()