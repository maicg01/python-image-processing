import cv2

def compress_and_remove_duplicate_frames(input_file, output_file):
    # Đọc video gốc
    video = cv2.VideoCapture(input_file)

    # Lấy thông số video gốc
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Tạo đối tượng VideoWriter để ghi video nén
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Khởi tạo biến lưu trữ khung hình trước đó
    prev_frame = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Kiểm tra khung hình hiện tại có giống khung hình trước đó không
        if prev_frame is not None and (frame == prev_frame).all():
            continue  # Bỏ qua nếu khung hình trùng lặp

        # Nén và ghi khung hình vào video nén
        output.write(frame)

        # Cập nhật khung hình trước đó
        prev_frame = frame.copy()

    # Giải phóng tài nguyên
    video.release()
    output.release()
    cv2.destroyAllWindows()

# Sử dụng hàm để nén và loại bỏ khung hình giống nhau
compress_and_remove_duplicate_frames('/home/maicg/Downloads/toan_canh_hop_quoc_hoi_ngay_27_5_ky_hop_thu_5_quoc_hoi_khoa_xv_vnews.mp4', 'output.mp4')