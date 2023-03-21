import cv2

# Đọc ảnh
img = cv2.imread('/home/maicg/Documents/Me/python-image-processing/code-edit-insightFace/quality-image/dataMe/dataTest/quality_result_bad/frame1786_0.1874.jpg')

# Áp dụng lọc Gaussian với kernel size là 5
filtered = cv2.GaussianBlur(img, (5, 5), 0)

# Tách các kênh màu
b, g, r = cv2.split(filtered)

# Ghép các kênh màu lại với nhau
filtered_color = cv2.merge((b, g, r))

# Hiển thị ảnh gốc và ảnh đã được lọc Gaussian
cv2.imshow('Original', img)
cv2.imshow('Filtered', filtered_color)

cv2.waitKey(7000)
cv2.destroyAllWindows()