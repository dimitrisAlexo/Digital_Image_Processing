import heapq
import sys
import numpy as np
import cv2
import skimage

np.set_printoptions(threshold=sys.maxsize)


def get_contour(x):
    # Finds the contour (one or more) of a letter by taking the dilated image and subtracting from the original. Then
    # it performs thinning of the boarders and detects how many contours the letter has.
    # - x: input image containing one letter
    # - c: cell array with one cell for each contour of the letter; each cell contains an N x 2 matrix, where N is
    # the number of points that describe the contour and each row has the two coordinates of each point

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow("img", x)
    # cv2.waitKey(0)

    # Resize and normalize the image
    max_width = 110
    max_height = 110
    x = cv2.resize(x, (110, 110))
    # x = cv2.resize(x, (0, 0), fx=min(max_width / x.shape[1], max_height / x.shape[0]),
    #                fy=min(max_width / x.shape[1], max_height / x.shape[0]))
    x = cv2.GaussianBlur(x, (5, 5), 0)
    x = cv2.normalize(x, None, -200, 500, cv2.NORM_MINMAX)

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow("img", x)
    # cv2.waitKey(0)

    # Convert the image to grayscale
    if len(x.shape) == 2 or x.shape[2] == 1:
        # Image is already grayscale, no need to convert it
        grayscale = x
    else:
        grayscale = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # Get the dilated image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(grayscale, kernel, iterations=1)

    # Subtract the dilated image from the original
    subtracted = cv2.subtract(dilated, grayscale)

    # Make image binary
    ret, binary = cv2.threshold(subtracted, 35, 255, 0)

    thinned = skimage.morphology.skeletonize(binary)
    thinned = np.asarray(thinned, dtype="uint8")
    thinned = thinned * 255
    thinned = cv2.bitwise_not(thinned)

    # # Print the contour
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow("img", thinned)
    # cv2.waitKey(0)

    # print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in thinned]))

    # thinned = [[255, 0, 0, 0, 255], [0, 255, 255, 255, 0], [0, 255, 0, 255, 0], [0, 255, 255, 255, 0],
    #            [0, 255, 0, 255, 0], [0, 255, 255, 255, 0], [255, 0, 0, 0, 255]]
    # thinned = np.array(thinned)

    # Find the coordinates of all black pixels
    black_pixels = np.argwhere(thinned == 0)

    # Initialize variables for tracking current contour
    current_contour = []
    c = []
    filler = []

    # Start at top left corner
    current_pixel = black_pixels[0]

    # Add first pixel to outer contour
    current_contour.append(list(current_pixel))

    count = 0

    # Loop until we've come back to starting point
    while True:

        # Define search area around current pixel
        min_row = max(0, current_pixel[0] - 1)
        max_row = min(thinned.shape[0] - 1, current_pixel[0] + 1)
        min_col = max(0, current_pixel[1] - 1)
        max_col = min(thinned.shape[1] - 1, current_pixel[1] + 1)

        # Find all neighboring black pixels in search area
        neighbors = [list((i, j)) for i in range(min_row, max_row + 1) for j in range(min_col, max_col + 1)
                     if thinned[i, j] == 0 and list((i, j)) not in current_contour]

        if not neighbors:
            count += 1
            if count == 4:
                smallest_elements = heapq.nsmallest(2, c, key=len)
                c = [item for item in c if item not in smallest_elements]
                break
            if len(current_contour) > 10:
                c.append(current_contour.copy())
                filler.append(current_contour.copy())
            else:
                filler.append(current_contour.copy())
                # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                # cv2.imshow("img", thinned)
                # cv2.waitKey(0)
            current_pixel = None
            for pixel in black_pixels:
                if not any(list(pixel) in sublist for sublist in filler):
                    current_pixel = list(pixel)
                    current_contour.clear()
                    current_contour.append(current_pixel)
                    break
            if current_pixel:
                continue
            break

        next_pixel = min(neighbors, key=lambda p: (p[0], -p[1]))
        current_pixel = next_pixel
        current_contour.append(list(current_pixel))

    # # Create binary image with only the pixels indicated by the outer and inner contours
    # binary_contours = np.zeros_like(thinned) + 255
    # for pixel in c[0]:
    #     binary_contours[pixel[0], pixel[1]] = 0
    # if len(c) > 1:
    #     for pixel in c[1]:
    #         binary_contours[pixel[0], pixel[1]] = 200
    # if len(c) > 2:
    #     for pixel in c[2]:
    #         binary_contours[pixel[0], pixel[1]] = 150
    #
    # binary_contours = binary_contours.astype(np.uint8)
    #
    # # Show the binary image with only the pixels indicated by the contour
    # cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    # cv2.imshow('Contours', binary_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return c
