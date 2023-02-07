import cv2
import matplotlib.pyplot as plt
import numpy as np

# image_src = cv2.imread('/home/maicg/Documents/Me/test-face-recognition/A_quality/8_2/frame195_0.0.jpg')
# # image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
# image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
# image_file = '/home/maicg/Documents/Me/test-face-recognition/A_quality/8_2/frame195_0.0.jpg'
image_file = '/home/maicg/Documents/Me/test-face-recognition/A_quality/full_data/frame1376_0.0255.jpg' #full
def read_this(image_file, gray_scale=False): 
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def his_Equalization(image_file):
    img = read_this(image_file)
    
    # Chuyển sang kênh màu đỏ
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]

    # Histogram Equalization cho kênh màu đỏ
    red = cv2.equalizeHist(red)
    green = cv2.equalizeHist(green)
    blue = cv2.equalizeHist(blue)

    # Chuyển trở lại kênh màu RGB
    img[:,:,0] = red
    img[:,:,1] = green
    img[:,:,2] = blue

    plt.imshow(img)
    plt.show()

def denoising(image_file):
    img = read_this(image_file)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 41)
    plt.imshow(img)
    plt.show()

def Brightness_Contrast(image_file, alpha, beta): #alpha: do tuong phan, beta: do sang
    img = read_this(image_file)
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    plt.imshow(img)
    plt.show()

def Sharpening(image_file):
    img = read_this(image_file)
    img = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0)
    plt.imshow(img)
    plt.show()

def Wiener_filter(image_file):
    img = read_this(image_file)
    # Tách các kênh màu (R, G, B)
    r, g, b = cv2.split(img)

    # # Tạo ma trận lọc Wiener
    # wiener_r = cv2.createWienerFilter(r.shape, 10, noiseVariance=0.1)
    # wiener_g = cv2.createWienerFilter(g.shape, 10, noiseVariance=0.1)
    # wiener_b = cv2.createWienerFilter(b.shape, 10, noiseVariance=0.1)

    # # Áp dụng Wiener filter cho từng kênh màu
    # result_r = cv2.filter2D(r, -1, wiener_b)
    # result_g = cv2.filter2D(g, -1, wiener_g)
    # result_b = cv2.filter2D(b, -1, wiener_r)

    # # Ghép các kênh màu thành hình ảnh màu
    # result = cv2.merge((result_r, result_g, result_b))

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Hiển thị kết quả
    plt.imshow(dst)
    plt.show()


# his_Equalization(image_file)
# denoising(image_file)
# Brightness_Contrast(image_file, 1.5, 10)
# Sharpening(image_file)
Wiener_filter(image_file)