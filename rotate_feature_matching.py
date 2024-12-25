
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform, warp
from skimage import img_as_ubyte

# Đọc ảnh gốc và ảnh bị biến dạng
original = cv2.imread('/home/lab-00/Downloads/orin_1.jpg', cv2.IMREAD_GRAYSCALE)
# original = cv2.imread('/home/lab-00/Downloads/orin.jpeg', cv2.IMREAD_GRAYSCALE)
distorted = cv2.imread('/home/lab-00/Downloads/quay.jpg', cv2.IMREAD_GRAYSCALE)

# Khởi tạo SURF detector (cần opencv-contrib-python để sử dụng SURF)
surf = cv2.xfeatures2d.SURF_create(400)

# Phát hiện các keypoints và descriptors cho cả hai ảnh
kp_original, des_original = surf.detectAndCompute(original, None)
kp_distorted, des_distorted = surf.detectAndCompute(distorted, None)

# Khởi tạo BFMatcher để so sánh các descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Tìm các cặp điểm khớp giữa các descriptors
matches = bf.match(des_original, des_distorted)

good = []
for m in matches:
    # print("matchereee: ", m.queryIdx, m.trainIdx)
    pt1 = kp_original[m.queryIdx].pt
    pt2 = kp_distorted[m.trainIdx].pt
    # if (m.distance < 20 and 
    #     abs(pt1[1] - pt2[1]) < 30 and 
    #     abs(pt1[0] - pt2[0]) < 50):   # epipolar constraint
    if m.distance < 5:
        good.append(m)

# Sắp xếp các matches theo khoảng cách
matches = sorted(good, key=lambda x: x.distance)

# Chuyển đổi các điểm matched
matched_original_pts = np.array([kp_original[m.queryIdx].pt for m in matches])
matched_distorted_pts = np.array([kp_distorted[m.trainIdx].pt for m in matches])

# Hiển thị các điểm matched giữa ảnh gốc và ảnh bị biến dạng
matched_img = cv2.drawMatches(original, kp_original, distorted, kp_distorted, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.imshow(matched_img)
plt.title("Putatively matched points (including outliers)")
plt.show()

# Tính toán ma trận biến đổi (geometric transformation)
tform = SimilarityTransform()
inlier_idx = tform.estimate(matched_distorted_pts, matched_original_pts)

# Inliers
inlier_distorted = matched_distorted_pts[inlier_idx]
inlier_original = matched_original_pts[inlier_idx]

# Hiển thị các điểm inliers
fig, ax = plt.subplots()
ax.set_title("Matching points (inliers only)")
# ax.imshow(np.hstack([original, distorted]), cmap='gray')

# Hiển thị các điểm inliers
ax.plot(inlier_original[:, 0], inlier_original[:, 1], 'go', label="ptsOriginal")
ax.plot(inlier_distorted[:, 0] + original.shape[1], inlier_distorted[:, 1], 'ro', label="ptsDistorted")
ax.legend()
plt.show()

# Tính toán ma trận ngược (inverse transformation)
A_inv = np.linalg.inv(tform.params)

# Tính toán tỷ lệ phục hồi (scale)
ss = A_inv[0, 1]
sc = A_inv[0, 0]
scale_recovered = np.hypot(ss, sc)
print(f"Recovered scale: {scale_recovered}")

# Tính toán góc quay phục hồi
theta_recovered = np.degrees(np.arctan2(-ss, sc))
print(f"Recovered theta: {theta_recovered}")

# Hiển thị kết quả
print(f"Scale: {tform.scale}")
print(f"RotationAngle: {tform.rotation}")

# Áp dụng biến đổi ngược cho ảnh bị biến dạng
output_view = np.zeros_like(original)
recovered = warp(distorted, tform.inverse, output_shape=original.shape)
recovered = img_as_ubyte(recovered)

# Hiển thị ảnh gốc và ảnh đã phục hồi
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(recovered, cmap='gray')
axes[1].set_title('Recovered Image')
plt.show()
