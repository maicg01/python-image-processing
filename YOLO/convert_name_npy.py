import numpy as np

# Đường dẫn đến file .npy ban đầu
input_file_path = "/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/database_local/cerberus_ada_r50.npy"

# Đọc dữ liệu từ file .npy
names = np.load(input_file_path)

# Chuyển đổi tên
converted_names = []
for name in names:
    words = name.split("-")
    converted_words = [word.capitalize() for word in words]
    converted_name = "".join(converted_words)
    # converted_name = "D/c " + name
    print(converted_name)
    converted_names.append(converted_name)

# Đường dẫn đến file .npy mới
output_file_path = "ten_vi_converted_new.npy"

# Lưu danh sách tên đã chuyển đổi vào file .npy mới
np.save(output_file_path, converted_names)

# In thông báo khi hoàn thành
print("Đã lưu danh sách tên đã chuyển đổi vào file .npy mới.")