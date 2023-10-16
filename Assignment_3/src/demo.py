from helpers import *
from harrisDetector import *
from matchDescriptor import *
from ransac import *
from imageStitching import *
import time

start_time = time.time()

print("--------------------")
print("City Region")
print("--------------------")

img1 = cv2.imread("im1.png")
img2 = cv2.imread("im2.png")
img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

p = [200, 200]
d = my_local_descriptor(img1, p, 5, 20, 1, 8)
print("Local descriptor @[200, 200]: ", d)
d = my_local_descriptor_upgrade(img1, p, 5, 20, 1, 8, 8)
print("Local descriptor upgrade @[200, 200]: ", d)

corners1 = detect_harris_features(img1_g, 0.6)
save_image_with_corners(img1.copy(), corners1, "Corners_City1.png")

corners2 = detect_harris_features(img2_g, 0.6)
save_image_with_corners(img2.copy(), corners2, "Corners_City2.png")

matching_points = descriptor_matching(corners1, corners2, img1_g, img2_g, 30)

r = 60
N = 10000
H, inlier_matching_points, outlier_matching_points = my_RANSAC(matching_points, r, N)

print("H transform: ", H)

img1_liers = img1.copy()
img2_liers = img2.copy()

for outlier in outlier_matching_points:
    pair_1, pair_2 = outlier

    im1_y1, im1_x1 = pair_1
    im2_y1, im2_x1 = pair_1
    im1_y2, im1_x2 = pair_2
    im2_y2, im2_x2 = pair_2

    color = (128, 128, 128)  # Gray

    thickness = -1  # Negative thickness for filled shape
    size = 10
    cv2.rectangle(img1_liers, (im1_x1-size, im1_y1-size), (im1_x1+size, im1_y1+size), color, thickness)
    cv2.rectangle(img1_liers, (im1_x2-size, im1_y2-size), (im1_x2+size, im1_y2+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x1-size, im2_y1-size), (im2_x1+size, im2_y1+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x2-size, im2_y2-size), (im2_x2+size, im2_y2+size), color, thickness)

for inlier in inlier_matching_points:
    pair_1, pair_2 = inlier

    im1_y1, im1_x1 = pair_1
    im2_y1, im2_x1 = pair_1
    im1_y2, im1_x2 = pair_2
    im2_y2, im2_x2 = pair_2

    color = np.random.randint(0, 256, size=3).tolist()  # Generate a random RGB color

    thickness = -1  # Negative thickness for filled shape
    size = 10
    cv2.rectangle(img1_liers, (im1_x1-size, im1_y1-size), (im1_x1+size, im1_y1+size), color, thickness)
    cv2.rectangle(img1_liers, (im1_x2-size, im1_y2-size), (im1_x2+size, im1_y2+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x1-size, im2_y1-size), (im2_x1+size, im2_y1+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x2-size, im2_y2-size), (im2_x2+size, im2_y2+size), color, thickness)

cv2.imwrite("City_Liers1.png", img1_liers)
cv2.imwrite("City_Liers2.png", img2_liers)

stitched = my_stitch(img1, img2, H)
cv2.imwrite("City_STITCHED.png", stitched)

# Filter for pepper and salt noise
ksize = 3
stitched_filtered = cv2.medianBlur(stitched, ksize)

cv2.imwrite("City_STITCHED_FILTERED.png", stitched_filtered)

print("--------------------")
print("Forest Region")
print("--------------------")

img1 = cv2.imread("imForest1.png")
img2 = cv2.imread("imForest2.png")
img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

p = [200, 200]
d = my_local_descriptor(img1, p, 5, 20, 1, 8)
print("Local descriptor @[200, 200]: ", d)
d = my_local_descriptor_upgrade(img1, p, 5, 20, 1, 8, 8)
print("Local descriptor upgrade @[200, 200]: ", d)

corners1 = detect_harris_features(img1_g, 0.15)
save_image_with_corners(img1.copy(), corners1, "Corners_Forest1.png")

corners2 = detect_harris_features(img2_g, 0.15)
save_image_with_corners(img2.copy(), corners2, "Corners_Forest2.png")

matching_points = descriptor_matching(corners1, corners2, img1_g, img2_g, 30)

r = 60
N = 10000
H, inlier_matching_points, outlier_matching_points = my_RANSAC(matching_points, r, N)

print("H transform: ", H)

img1_liers = img1.copy()
img2_liers = img2.copy()

for outlier in outlier_matching_points:
    pair_1, pair_2 = outlier

    im1_y1, im1_x1 = pair_1
    im2_y1, im2_x1 = pair_1
    im1_y2, im1_x2 = pair_2
    im2_y2, im2_x2 = pair_2

    color = (128, 128, 128)  # Gray

    thickness = -1  # Negative thickness for filled shape
    size = 10
    cv2.rectangle(img1_liers, (im1_x1-size, im1_y1-size), (im1_x1+size, im1_y1+size), color, thickness)
    cv2.rectangle(img1_liers, (im1_x2-size, im1_y2-size), (im1_x2+size, im1_y2+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x1-size, im2_y1-size), (im2_x1+size, im2_y1+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x2-size, im2_y2-size), (im2_x2+size, im2_y2+size), color, thickness)

for inlier in inlier_matching_points:
    pair_1, pair_2 = inlier

    im1_y1, im1_x1 = pair_1
    im2_y1, im2_x1 = pair_1
    im1_y2, im1_x2 = pair_2
    im2_y2, im2_x2 = pair_2

    color = np.random.randint(0, 256, size=3).tolist()  # Generate a random RGB color

    thickness = -1  # Negative thickness for filled shape
    size = 10
    cv2.rectangle(img1_liers, (im1_x1-size, im1_y1-size), (im1_x1+size, im1_y1+size), color, thickness)
    cv2.rectangle(img1_liers, (im1_x2-size, im1_y2-size), (im1_x2+size, im1_y2+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x1-size, im2_y1-size), (im2_x1+size, im2_y1+size), color, thickness)
    cv2.rectangle(img2_liers, (im2_x2-size, im2_y2-size), (im2_x2+size, im2_y2+size), color, thickness)

cv2.imwrite("Forest_Liers1.png", img1_liers)
cv2.imwrite("Forest_Liers2.png", img2_liers)

stitched = my_stitch(img1, img2, H)
cv2.imwrite("Forest_STITCHED.png", stitched)

# Filter for pepper and salt noise
ksize = 3
stitched_filtered = cv2.medianBlur(stitched, ksize)

cv2.imwrite("Forest_STITCHED_FILTERED.png", stitched_filtered)

print("Execution time:", time.time() - start_time, "sec")
