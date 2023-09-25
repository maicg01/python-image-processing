import cv2

def detect_skin(image):
    # Chuyển đổi hình ảnh sang không gian màu YCrCb
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Định nghĩa khoảng màu sắc da trong không gian màu YCrCb
    lower_skin = (0, 120, 77)
    upper_skin = (255, 200, 150)

    # Tạo mặt nạ (mask) từ khoảng màu da
    skin_mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)

    # Áp dụng phép toán morp để loại bỏ các nhiễu nhỏ và kết nối các vùng da lại với nhau
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Áp dụng mặt nạ da lên hình ảnh gốc
    skin_detected_image = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin_detected_image


# Load input image
input_image = cv2.imread('/home/maicg/Downloads/c17d4dfd96464c181557.jpg')

# Apply skin color detection
skin_detected_image = detect_skin(input_image)


# Display the output image
cv2.imshow('Face Detection', skin_detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()