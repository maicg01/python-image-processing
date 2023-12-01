import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('.'))
print("sys.path", sys.path)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 100
img_template = cv2.imread('image/template.jpg')
img_need_aligned = cv2.imread(sys.argv[1])

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