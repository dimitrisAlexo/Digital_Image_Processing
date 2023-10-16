import numpy as np


def my_stitch(img1, img2, H):

    theta = H['theta']
    d = H['d']

    # Convert images to double precision
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    # Get image sizes
    M1, N1, _ = img1.shape
    M2, N2, _ = img2.shape

    # Define transformation parameters
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Create stitched image
    stitched_width = 3 * max(M1, M2)
    stitched_height = 3 * max(N1, N2)

    stitched = np.zeros((stitched_width, stitched_height, 3))

    # Calculate starting position
    start = np.ceil([stitched_width / 2, stitched_height / 2]).astype(int)

    # Stitch the images
    for i in range(start[0], stitched_width):
        x = i - start[0] + 1
        for j in range(start[1], stitched_height):
            y = j - start[1] + 1
            if x < M1 and y < N1:
                p = np.ceil(np.dot(R, [x, y]) + d).astype(int)
                stitched[start[0] + p[0], start[1] + p[1]] = img1[x - 1, y - 1]
            if x < M2 and y < N2:
                stitched[i, j] = img2[x - 1, y - 1]

    # Convert image to 8-bit unsigned integer
    stitched = (stitched * 255).astype(np.uint8)

    return stitched

