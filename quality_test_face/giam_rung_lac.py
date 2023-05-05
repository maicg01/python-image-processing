import cv2

# Tạo đối tượng VideoCapture để đọc video từ file
cap = cv2.VideoCapture('rtsp://ai_dev:123654789@@@192.168.15.10:554/Streaming/Channels/501')

# Lấy kích thước khung hình của video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Tạo đối tượng VideoWriter để ghi video đã xử lý vào file
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Tạo đối tượng ổn định khung hình
# stabilizer = cv2.DISOpticalFlow_create()

# Thiết lập kernel cho lọc thông thấp
kernel_size = 5
kernel = cv2.getGaussianKernel(kernel_size, 0)

# Đọc từng khung hình từ video và xử lý
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ổn định khung hình
    # stabilized_frame = stabilizer.calc(frame)

    # Giảm nhiễu
    denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Lọc thông thấp
    filtered_frame = cv2.filter2D(denoised_frame, -1, kernel)

    # Ghi khung hình đã xử lý vào video đầu ra
    # out.write(filtered_frame)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('Filtered Frame', filtered_frame)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ
cap.release()
out.release()
cv2.destroyAllWindows()