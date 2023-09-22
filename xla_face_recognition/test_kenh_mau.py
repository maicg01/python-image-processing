import cv2

# Đọc hình ảnh
image = cv2.imread("/home/maicg/Downloads/face_NGUYENNH/person1/no_person1_11_44_43_17_08_2023.jpg")

# Kiểm tra kích thước của hình ảnh
height, width, channels = image.shape

# Kiểm tra số lượng kênh màu
if channels == 3:
    # Nếu có 3 kênh màu, kiểm tra xem đó là RGB hay BGR
    if image[:,:,0].mean() > image[:,:,2].mean():
        print("Hình ảnh có định dạng RGB")
    else:
        print("Hình ảnh có định dạng BGR")
else:
    print("Hình ảnh không có định dạng RGB hoặc BGR")