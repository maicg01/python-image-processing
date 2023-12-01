import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('.'))
print("sys.path", sys.path)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
img_template = cv2.imread(sys.argv[1])
img_need_aligned = cv2.imread(sys.argv[2])

im1Gray = cv2.cvtColor(img_need_aligned, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(img_need_aligned, keypoints1, img_template, keypoints2, matches, None)
cv2.imwrite("matches.jpg", imMatches)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
# Use homography
height, width, channels = img_template.shape
im1Reg = cv2.warpPerspective(img_need_aligned, h, (width, height))

cv2.imshow('Face Align', im1Reg)
cv2.waitKey()

# # cach 2
# import os

# import cv2
# import numpy as np

# from config import folder_path_aligned_images

# MAX_FEATURES = 500
# GOOD_MATCH_PERCENT = 0.15


# class OpenCV:
#     @classmethod
#     def match_img(cls, im1, im2):
#         # Convert images to grayscale
#         im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#         im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#         # Detect ORB features and compute descriptors.
#         orb = cv2.ORB_create(MAX_FEATURES)
#         keypoints_1, descriptors_1 = orb.detectAndCompute(im1_gray, None)
#         keypoints_2, descriptors_2 = orb.detectAndCompute(im2_gray, None)

#         # Match features.
#         matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#         matches = matcher.match(descriptors_1, descriptors_2, None)

#         # Sort matches by score
#         matches.sort(key=lambda x: x.distance, reverse=False)

#         # Remove not so good matches
#         num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
#         matches = matches[:num_good_matches]

#         # Draw top matches
#         im_matches = cv2.drawMatches(im1, keypoints_1, im2, keypoints_2, matches, None)
#         cv2.imwrite(os.path.join(folder_path_aligned_images, "matches.jpg"), im_matches)

#         # Extract location of good matches
#         points_1 = np.zeros((len(matches), 2), dtype=np.float32)
#         points_2 = np.zeros((len(matches), 2), dtype=np.float32)

#         for i, match in enumerate(matches):
#             points_1[i, :] = keypoints_1[match.queryIdx].pt
#             points_2[i, :] = keypoints_2[match.trainIdx].pt

#         # Find homography
#         h, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC)

#         # Use homography
#         height, width, channels = im2.shape
#         im1_reg = cv2.warpPerspective(im1, h, (width, height))

#         return im1_reg, h

#     @classmethod
#     def align_img(cls, template_path, raw_img_path, result_img_path):
#         # Read reference image
#         ref_filename = template_path
#         print("Reading reference image: ", ref_filename)
#         im_reference = cv2.imread(ref_filename, cv2.IMREAD_COLOR)

#         # Read image to be aligned
#         im_filename = raw_img_path
#         print("Reading image to align: ", im_filename)
#         im = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)

#         print("Aligning images ...")
#         # Registered image will be resorted in im_reg.
#         im_reg, h = OpenCV.match_img(im, im_reference)

#         # Write aligned image to disk.
#         print("Saving aligned image : ", result_img_path)
#         cv2.imwrite(result_img_path, im_reg)

#         return result_img_path