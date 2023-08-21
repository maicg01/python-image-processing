import cv2
import os

def extract_frames(video_path, output_path, num_frames=15):
    # Kiểm tra nếu thư mục đầu ra không tồn tại, tạo mới nếu cần
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Đọc video
    video = cv2.VideoCapture(video_path)

    # Tính tổng số frame trong video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính bước nhảy giữa các frame để chọn ra 15 frame đều đặn
    step = max(total_frames // num_frames, 1)

    # Đọc và lưu các frame được chọn
    for i in range(0, total_frames, step):
        # Đọc frame
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()

        if ret:
            # Lưu frame vào thư mục đầu ra
            output_file = os.path.join(output_path, f"frame_{i}.jpg")
            print(output_file)
            cv2.imwrite(output_file, frame)

    # Giải phóng tài nguyên
    video.release()

# Đường dẫn tới thư mục chứa video
video_folder = r"C:\Users\nguye\Downloads\dataFaceRecog\video"
# Đường dẫn tới thư mục lưu frame
output_folder = r"C:\Users\nguye\Downloads\dataFaceRecog\save_video_dai_bieu"

# Lặp qua tất cả các file trong thư mục video
for filename in os.listdir(video_folder):
    if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".MOV") or filename.endswith(".mov"):
        video_path = os.path.join(video_folder, filename)
        output_path = os.path.join(output_folder, filename.split(".")[0])
        extract_frames(video_path, output_path)