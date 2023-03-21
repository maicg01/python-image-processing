import cv2
import numpy as np

# Đọc các ảnh đầu vào
image1 = cv2.imread("/home/maicg/Documents/Me/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataTest/quality_result_bad/frame2057_0.1626.jpg")
image2 = cv2.imread("/home/maicg/Documents/Me/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataTest/quality_result_bad/frame1667_0.0516.jpg")

# Tách các kênh màu của ảnh
b1, g1, r1 = cv2.split(image1)
b2, g2, r2 = cv2.split(image2)

# Tạo mặt nạ (mask) từ các kênh màu
mask_b = cv2.absdiff(b1, b2)
mask_g = cv2.absdiff(g1, g2)
mask_r = cv2.absdiff(r1, r2)

# Áp dụng Gaussian Blur để loại bỏ nhiễu
mask_b = cv2.GaussianBlur(mask_b, (5, 5), 0)
mask_g = cv2.GaussianBlur(mask_g, (5, 5), 0)
mask_r = cv2.GaussianBlur(mask_r, (5, 5), 0)

# Chuyển đổi mặt nạ thành ảnh nhị phân
ret, mask_b = cv2.threshold(mask_b, 20, 255, cv2.THRESH_BINARY)
ret, mask_g = cv2.threshold(mask_g, 20, 255, cv2.THRESH_BINARY)
ret, mask_r = cv2.threshold(mask_r, 20, 255, cv2.THRESH_BINARY)

# Kết hợp các kênh màu để tạo ra ảnh mới
image_new = cv2.merge((b1 * (mask_b == 255) + b2 * (mask_b == 0),
                       g1 * (mask_g == 255) + g2 * (mask_g == 0),
                       r1 * (mask_r == 255) + r2 * (mask_r == 0)))

# Hiển thị ảnh sau tái tạo
cv2.imshow("Reconstructed Image", image_new)
cv2.waitKey(3000)
cv2.destroyAllWindows()