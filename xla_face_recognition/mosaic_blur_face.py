import cv2

def mosaic_blur(image, block_size):
    # Kích thước ảnh
    height, width, _ = image.shape

    # Chia nhỏ ảnh thành các khối
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Lấy giá trị màu trung bình của khối
            block = image[y:y+block_size, x:x+block_size]
            avg_color = cv2.mean(block)[:3]

            # Đặt giá trị màu trung bình cho toàn bộ khối
            image[y:y+block_size, x:x+block_size] = avg_color

    return image

# Đường dẫn đến ảnh
image_path = "/home/maicg/Desktop/test.mai.nguyen.augu/test.mai.nguyen/1.jpg"

# Đọc ảnh gốc
image = cv2.imread(image_path)

# Thiết lập kích thước khối mosaic
for i in range(10,21,2):
    image_blur = image.copy()
    block_size = i

    # Áp dụng hiệu ứng mosaic blur
    result = mosaic_blur(image_blur, block_size)
    path_save = "/home/maicg/Desktop/test.mai.nguyen.augu/test.mai.nguyen/1" + "_blur_mosaic_" + str(i) + ".jpg"
    cv2.imwrite(path_save, result)

    # Hiển thị ảnh gốc và kết quả
    cv2.imshow("Original Image", image)
    cv2.imshow("Mosaic Blur", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()