import numpy as np
import sys

# file npy
input_file_path = sys.argv[1]
ten_ID = sys.argv[2]

# Đọc dữ liệu từ file .npy
names = np.load(input_file_path)

# Chuyển đổi tên
converted_names = []
for name in names:
    # if ten_ID == name:
        converted_names.append(name)

print(converted_names)
print(len(converted_names))