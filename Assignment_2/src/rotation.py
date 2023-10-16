import imutils as imutils
import numpy as np
import cv2


def find_rotation_angle(x):
    # Finds the angle a text image might have been rotated. First blurs the image so that the letters of each line
    # attach with each other and then takes the logarithm of the magnitude of the image's DFT. Based on the maximum
    # frequency, which corresponds to the difference in brightness from one line to the next, it calculates the angle
    # the image has been rotated in regard to the x-axis
    # - x: the input image
    # - angle: the angle the text image might have been rotated

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(grayscale, (15, 15), 0)

    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)

    # calculate the magnitude of the Fourier Transform
    magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

    # Remove the DC
    height, width = magnitude.shape[:2]
    (centerX, centerY) = (width // 2, height // 2)
    center = (centerX, centerY)
    radius = 5
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    magnitude[mask == 255] = 0
    # Remove high frequencies
    low_radius = 200
    low_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(low_mask, center, low_radius, 1, -1)
    magnitude *= low_mask

    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, -255, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # # Display the magnitude of the Fourier Transform
    # cv2.namedWindow('magnitude', cv2.WINDOW_NORMAL)
    # cv2.imshow('magnitude', magnitude.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    height, width = magnitude.shape[:2]
    (centerX, centerY) = (width // 2, height // 2)
    right_half = magnitude[:, centerX:]
    max_row, max_col = np.unravel_index(np.argmax(right_half), right_half.shape)
    max_col += centerX

    if max_row == centerY:
        angle = 90.0
    elif max_col == centerX:
        angle = 0.0
    else:
        angle = np.arctan((max_row - centerY) / (max_col - centerX))
        if angle >= 0:
            angle = 90.0 - np.rad2deg(angle)
        else:
            angle = - np.rad2deg(angle) - 90.0

    best_angle = angle
    best_gradient = -np.inf
    for delta in np.arange(-1, 1, 0.05):
        rotated = rotate_image(x, -angle + delta)
        projection = np.sum(rotated, axis=1)
        gradient = np.gradient(projection)
        max_gradient = np.sum(np.abs(gradient))
        if max_gradient > best_gradient:
            best_gradient = max_gradient
            best_angle = angle - delta
    angle = best_angle

    return angle


def rotate_image(x, angle):
    # Rotates an image x at an angle (positive or negative). The spaces are filled using the appropriate padding.
    # - x: the input image
    # - angle: the angle we want to rotate the image
    # - y: the output image

    # Get image height and width
    height, width = x.shape[:2]

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Compute the sine and cosine of the angle
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Compute the new image dimensions
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Compute the translation matrix to center the image
    M[0, 2] += (new_width - width) / 2
    M[1, 2] += (new_height - height) / 2

    # Perform the rotation with appropriate padding using linear interpolation
    y = cv2.warpAffine(x, M, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(255, 255, 255))

    return y
