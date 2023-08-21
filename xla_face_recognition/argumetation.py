import cv2
import numpy as np

def augment_saturation(image):
    # Tính toán các giá trị mới cho các yếu tố argumentation
    saturation_factor = 0.25  
    # Saturation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    augmented = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return augmented

def augment_blur(image):
    blur_kernel_size =5
    augmented = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    return augmented

def augment_noise(image):
    noise_std_dev = np.random.uniform(0, 0.1)
    noise = np.random.normal(0, noise_std_dev, image.shape).astype(np.uint8)
    augmented = np.clip(image + noise, 0, 255)
    return augmented

# Đường dẫn đến ảnh cần argumentation
image_path = "/home/maicg/Downloads/dataFacerecog/detect/b121/0.jpg"

# Đọc ảnh
image = cv2.imread(image_path)

# Argumentation cho ảnh
img1 = augment_saturation(image)

# Argumentation cho ảnh
img2 = augment_blur(image)

# Argumentation cho ảnh
img3 = augment_noise(image)

# Hiển thị ảnh gốc và ảnh đã được argumentation
cv2.imshow("Original Image", image)
cv2.imshow("Augmented Image", img1)
cv2.imshow("Augmented Image2", img2)
cv2.imshow("Augmented Image3", img3)
cv2.waitKey(0)
# cv2.destroyAllWindows()