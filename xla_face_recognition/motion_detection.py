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

    # Tính toán sự khác biệt giữa khung hình hiện tại và khung hình trước đó
    frame_diff = cv2.absdiff(current_gray, previous_gray)

    # Áp dreeshold để tách các vùng chuyển động
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Kiểm tra xem có vùng chuyển động trong khung hình hay không
    has_motion = cv2.countNonZero(threshold) > 0

    # Áp dụng vùng chuyển động lên khung hình BGR
    # motion_visualize = cv2.bitwise_and(current_frame, current_frame, mask=threshold)

    if has_motion:
        # Hiển thị khung hình với vùng chuyển động
        cv2.imshow('Motion Detection', current_frame)

    # Cập nhật khung hình trước là khung hình hiện tại
    previous_gray = current_gray

    # Kiểm tra phím nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video.release()
cv2.destroyAllWindows()