import cv2


def rotate_image(x, angle):
    # Rotates an image x at an angle (positive or negative). The spaces are filled using the appropriate padding.
    # - x: the input image
    # - angle: the angle we want to rotate the image (in degrees)
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


def save_image_with_corners(img, corners, output_path):

    # Draw red circles at corner locations
    for corner in corners:
        y, x = corner
        cv2.circle(img, (x, y), 10, (0, 0, 255), 3)

    # Save the image with corners
    cv2.imwrite(output_path, img)
