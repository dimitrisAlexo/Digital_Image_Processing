import cv2
import numpy as np


def is_corner(img, points, k, Rthres):
    # Calculate image derivatives
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate products of derivatives
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy

    # Calculate sums of derivatives within the neighborhood
    dxx_sum = cv2.boxFilter(dxx, cv2.CV_64F, (3, 3))
    dyy_sum = cv2.boxFilter(dyy, cv2.CV_64F, (3, 3))
    dxy_sum = cv2.boxFilter(dxy, cv2.CV_64F, (3, 3))

    # Calculate Harris response
    det = dxx_sum * dyy_sum - dxy_sum * dxy_sum
    trace = dxx_sum + dyy_sum
    R = det - k * (trace ** 2)

    # Check if the Harris response is above the threshold for each point
    return [abs(R[p[0], p[1]]) > Rthres for p in points]


def my_detect_harris_features(img, min_distance=10):
    k = 0.04  # Harris detector parameter
    Rthres = 8421000 * np.max(img)  # Harris' response threshold

    height, width = img.shape

    # Generate a grid of points for the image
    grid_y, grid_x = np.mgrid[1:height-1, 1:width-1]
    points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

    # Evaluate the corner response for all points
    is_corner_list = is_corner(img, points, k, Rthres)

    # Extract corner coordinates where the response is True
    corners = points[np.array(is_corner_list)]

    # Filter out nearby points
    filtered_corners = []
    for corner in corners:
        if not any(np.linalg.norm(corner - filtered_corner) < min_distance for filtered_corner in filtered_corners):
            filtered_corners.append(corner)

    # Return the coordinates of the corners
    return np.array(filtered_corners)


def detect_harris_features(img, Rthres, min_distance=10):
    # Convert image to grayscale if it's not already
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Apply Harris corner detection
    corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)

    # Set a threshold to select strong corners
    threshold = Rthres * corners.max()
    corners = np.argwhere(corners > threshold)

    # Filter out nearby points
    filtered_corners = []
    for corner in corners:
        if not any(np.linalg.norm(corner - filtered_corner) < min_distance for filtered_corner in filtered_corners):
            filtered_corners.append(corner)

    # Return the coordinates of the corners
    return np.array(filtered_corners)
