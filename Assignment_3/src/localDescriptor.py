import cv2
import numpy as np


def my_local_descriptor(img, p, rhom, rhoM, rhostep, N):
    # Check if point p is too close to the image boundary
    h, w = img.shape[:2]
    if p[0] - rhoM < 0 or p[0] + rhoM >= w or p[1] - rhoM < 0 or p[1] + rhoM >= h:
        return []  # Return empty vector if p is too close to the boundary

    descriptor = []  # Initialize descriptor vector

    # Iterate over concentric circles
    for rho in np.arange(rhom, rhoM, rhostep):
        x_rho = []  # Vector for current circle

        # Iterate over points on the circle
        for i in range(N):
            angle = 2 * np.pi * i / N
            x = int(p[0] + rho * np.cos(angle))
            y = int(p[1] + rho * np.sin(angle))

            # Interpolate pixel value at (x, y)
            if 0 <= x < w and 0 <= y < h:
                x_interp = cv2.getRectSubPix(img, (1, 1), (x, y))[0][0]
            else:
                x_interp = 0  # Default value for pixels outside the image

            x_rho.append(x_interp)

        # Calculate the average value for the current circle
        avg_x_rho = np.mean(x_rho)
        descriptor.append(avg_x_rho)

    return descriptor


def my_local_descriptor_upgrade(img, p, rhom, rhoM, rhostep, N, num_bins):
    # Check if point p is too close to the image boundary
    h, w = img.shape[:2]
    if p[0] - rhoM < 0 or p[0] + rhoM >= w or p[1] - rhoM < 0 or p[1] + rhoM >= h:
        return []  # Return empty vector if p is too close to the boundary

    descriptor = []  # Initialize descriptor vector

    # Iterate over concentric circles
    for rho in np.arange(rhom, rhoM, rhostep):
        histogram = np.zeros(int(num_bins))  # Histogram for current circle

        # Iterate over points on the circle
        for i in range(N):
            angle = 2 * np.pi * i / N
            x = int(p[0] + rho * np.cos(angle))
            y = int(p[1] + rho * np.sin(angle))

            # Interpolate pixel value at (x, y)
            if 0 <= x < w and 0 <= y < h:
                x_interp = cv2.getRectSubPix(img, (1, 1), (x, y))[0][0]
            else:
                x_interp = 0  # Default value for pixels outside the image

            # Increment the corresponding histogram bin
            x_interp_scalar = np.mean(x_interp)
            bin_index = int((x_interp_scalar / 255) * (num_bins - 1))
            histogram[bin_index] += 1

        descriptor.extend(histogram)

    return descriptor

