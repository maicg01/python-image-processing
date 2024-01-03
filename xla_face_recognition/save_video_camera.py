import cv2
import sys


def save_camera_video(output_path, frame_rate=30.0):
    # Khởi tạo camera
    cap = cv2.VideoCapture(sys.argv[1])  # chỉ định sử dụng camera mặc định

    # Lấy kích thước khung hình từ camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video (ở đây là MP4)
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Đọc và ghi từng khung hình vào video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

        cv2.imshow('Video', frame)

        # Nhấn 'q' để dừng quá trình ghi video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Ví dụ sử dụng
output_path = 'output.mp4'  # Đường dẫn và tên file video đầu ra
frame_rate = 30.0  # Tốc độ khung hình (FPS)

# Gọi hàm save_camera_video để đọc khung hình từ camera và lưu thành video
save_camera_video(output_path, frame_rate)