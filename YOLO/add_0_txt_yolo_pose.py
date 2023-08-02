# them keypoint rong cho cac keypoit bi thieu cua yolo-pose
import os
# Mở file để đọc nội dung
file_txt = "/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/exp4/labels"
file_txt_save = "/home/maicg/Documents/Me/YOLO/yolov5/runs/detect/labels_face_pose"

for dir_txt in sorted(os.listdir(file_txt)):
    name_txt = os.path.join(file_txt,dir_txt)

    with open(name_txt, "r") as file:
        # Lưu nội dung file vào biến content
        content = file.readlines()

    name_txt_save = file_txt_save + "/" + dir_txt
    # Mở file để ghi
    with open(name_txt_save, "w") as file:
        # Duyệt qua từng dòng của nội dung file đã đọc được
        for line in content:
            # Thêm chuỗi "iloveyou" vào cuối dòng
            modified_line = line.strip() + " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
            # Ghi dòng đã được thêm chuỗi "iloveyou" vào file txt
            file.write(modified_line)