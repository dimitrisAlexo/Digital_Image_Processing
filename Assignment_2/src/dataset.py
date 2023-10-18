from rotation import *
from contour import *
from descriptor import *
import pandas as pd
import codecs


def get_dataset(img, text, blur, resize):
    # Forms a dataset by dividing the text image into its letters and putting them in an array. First reverses the image
    # rotation and then separates each line and each letter by taking the projection of brightness in the vertical and
    # horizontal axis respectively. Before putting each letter in the array it performs morphological "opening" to get
    # rid of any noise.
    # - img: the text image from which we want to extract the img_dataset
    # - text: the text document from which we want to extract the ascii_dataset
    # - img_dataset: a list containing the contour of each letter of the dataset
    # - ascii_dataset: a list containing each letter of the dataset in ascii code

    # Find the rotation angle of the image
    rotation_angle = find_rotation_angle(img)
    print("Image rotation: ", rotation_angle, "degrees")

    # Rotate the image to make the text horizontal
    img = rotate_image(img, -rotation_angle)

    # Replace the colored pixels with white
    color_indices = np.where(np.sum(img < 220, axis=2) < 3)
    img[color_indices] = 255

    # Resize image
    fx, fy = resize, resize
    img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)
    inverted = cv2.bitwise_not(grayscale)

    # Calculate the vertical projection of the image
    vertical_proj = np.sum(inverted, axis=1)

    # Find the indices where the projection changes from non-zero to zero
    line_indices = np.where(np.diff((vertical_proj == 0).astype(int)) == -1)[0] + 1

    # Split the image into lines
    lines = []
    offset_up = int(inverted.shape[0] / 250)
    offset_down = 0  # int(inverted.shape[0] / 100)
    for i in range(0, len(line_indices)):
        if i == 0:
            y1 = int(inverted.shape[0] / 250)
        else:
            y1 = line_indices[i - 1]
        y2 = line_indices[i]
        line = inverted[y1-offset_up:y2-offset_down, :]
        lines.append(line)
    end_line = line_indices[len(line_indices)-1]
    lines.append(inverted[end_line - offset_up:end_line + int(inverted.shape[0] / 30), :])

    # for line in lines:
    #     cv2.namedWindow('line', cv2.WINDOW_NORMAL)
    #     cv2.imshow('line', line.astype(np.uint8))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Initialize the dataset array
    words = []

    for line in lines:

        blurred_line = cv2.GaussianBlur(line, (blur * fx, blur * fy), 0)

        # Calculate the horizontal projection of the image
        horizontal_proj = np.sum(blurred_line, axis=0)

        # Find the indices where the projection first time goes under the threshold
        word_indices = np.where(np.diff((horizontal_proj > 0).astype(int)) == -1)[0] + 1

        line_words = []
        offset_left = 0
        offset_right = 0  # int(inverted.shape[1] / 300)
        for i in range(0, len(word_indices)):
            if i == 0:
                y1 = 0
            else:
                y1 = word_indices[i - 1]
            y2 = word_indices[i]
            word = line[:, y1 - offset_left:y2 + offset_right]
            line_words.append(word)

        words.append(line_words)

    # for line in words:
    #     for word in line:
    #         cv2.namedWindow('line', cv2.WINDOW_NORMAL)
    #         cv2.imshow('line', cv2.bitwise_not(word).astype(np.uint8))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    img_dataset = []

    for line in words:

        line_letters = []

        for word in line:

            # Remove pale pixels on the edges of the letters so that the letters do not overlap
            word[word < 135] = 0

            # Calculate the horizontal projection of the image
            horizontal_proj = np.sum(word, axis=0)

            # Find the indices where the projection first time goes under the threshold
            letter_indices = np.where(np.diff((horizontal_proj > 0).astype(int)) == -1)[0] + 1

            word_letters = []
            offset_left = 0
            offset_right = 5
            for i in range(0, len(letter_indices)):
                if i == 0:
                    y1 = 0
                else:
                    y1 = letter_indices[i - 1]
                y2 = letter_indices[i]
                letter = word[:, y1 - offset_left:y2 + offset_right]

                # cv2.namedWindow('line', cv2.WINDOW_NORMAL)
                # cv2.imshow('line', letter.astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                word_letters.append(letter)

            line_letters.append(word_letters)

        img_dataset.append(line_letters)

    # Calculate each contour for every letter of the text
    for line in img_dataset:
        for word in line:
            for i, letter in enumerate(word):

                letter = np.pad(letter, [(5, 0), (0, 0)], mode='constant')
                projection = np.sum(letter, axis=1)  # Compute column-wise sum of pixel values
                black_rows = np.where(projection == 0)[0]  # Get indices of black rows

                crop_position = np.argmax(
                    np.diff(black_rows) > 30) + 1  # Find position where consecutive black rows exceed 10
                if crop_position >= len(black_rows):
                    crop_position = len(black_rows) - 1

                crop_row = black_rows[crop_position] + 30  # Get the row to crop at
                letter = letter[:crop_row]  # Crop the image vertically

                if np.all(letter == 0):

                    height, width = letter.shape
                    center_x = width // 2
                    center_y = height // 2
                    letter[center_y-2:center_y+3, center_x-2:center_x+3] = 255

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                word[i] = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernel)
                word[i] = get_contour(word[i])

    # Try reading as UTF-8 first
    try:
        with codecs.open(text, encoding='utf-8') as f:
            lines = f.read().splitlines()
    # If that fails, try reading as UTF-16
    except UnicodeDecodeError:
        with codecs.open(text, encoding='utf-16') as f:
            lines = f.read().splitlines()

    ascii_dataset = []

    for line in lines:
        words = line.split(' ')
        word_list = []
        for word in words:
            if len(word) > 0:
                char_list = [str(char) for char in word]
                word_list.append(char_list)
        if len(word_list) > 0:
            ascii_dataset.append(word_list)

    print(ascii_dataset)
    print(sum(len(elem) for sublist in ascii_dataset for elem in sublist for elem in elem))
    print(sum(len(elem) for sublist in img_dataset for elem in sublist))

    print(ascii_dataset[4][5][1])

    return img_dataset, ascii_dataset


def divide_into_classes(img_dataset, ascii_dataset):

    # Divides the letters inside img_dataset into three classes, class1, class2, class3 depending on how many contours
    # each letter has. For every contour is puts into one of the three classes, it puts the corresponding ascii
    # character into the class as well.
    # - img_dataset: list of contours for every letter in every line and word of the text
    # - ascii_dataset: list of ascii characters for every letter in every line and word of the text
    # - class1, class2, class3: lists where the first elements are lists of the letters that belong to each class, and
    # the second elements are lists of the ascii characters that belong to each class as 1-1 match with the letters

    class1 = [[], []]
    class2 = [[], []]
    class3 = [[], []]

    for line in range(min(len(img_dataset), len(ascii_dataset))):
        for word in range(min(len(img_dataset[line]), len(ascii_dataset[line]))):
            for letter in range(min(len(img_dataset[line][word]), len(ascii_dataset[line][word]))):
                # Get the number of contours for the current letter
                num_contours = len(img_dataset[line][word][letter])
                # Append the letter and its ascii character to the corresponding class
                if num_contours == 1:
                    class1[0].append(img_dataset[line][word][letter])
                    class1[1].append(ascii_dataset[line][word][letter])
                elif num_contours == 2:
                    class2[0].append(img_dataset[line][word][letter])
                    class2[1].append(ascii_dataset[line][word][letter])
                elif num_contours == 3:
                    class3[0].append(img_dataset[line][word][letter])
                    class3[1].append(ascii_dataset[line][word][letter])

    print(class1[1])
    print(class2[1])
    print(class3[1])

    return class1, class2, class3


def form_dataset(class1, class2, class3, N):
    # Forms the final dataset that is to be passed to the KNN algorithm. Each row of the dataset is the descriptors for
    # each letter of the class plus the label of the letter. The label is just the ascii character that is stored in the
    # second list of the class.
    # - class1, class2, class3: the input classes each one containing a list of the letter contours and a list of the
    # corresponding ascii characters
    # - N: the size of the fixed-length descriptors
    # - dataset1, dataset2, dataset3: one dataset for each class as mentioned above

    dataset1 = pd.DataFrame()
    dataset2 = pd.DataFrame()
    dataset3 = pd.DataFrame()

    # process class1
    for i in range(len(class1[0])):
        letter = class1[0][i]
        label = class1[1][i]

        descriptor = get_descriptor(letter[0])

        # Interpolate descriptor so it is of fixed size
        interpolation_points = np.linspace(0, 1, N)
        descriptor = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor)), descriptor)

        row = np.concatenate((descriptor, [label]))  # add the label to the descriptor
        dataset1 = pd.concat([dataset1, pd.DataFrame([row])])

    # process class2
    for i in range(len(class2[0])):
        letter1, letter2 = class2[0][i]
        label = class2[1][i]

        descriptor1 = get_descriptor(letter1)
        descriptor2 = get_descriptor(letter2)

        # Interpolate descriptor so it is of fixed size
        interpolation_points = np.linspace(0, 1, N)
        descriptor1 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor1)), descriptor1)
        descriptor2 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor2)), descriptor2)

        row = np.concatenate((descriptor1, descriptor2, [label]))  # add the label and the two descriptors
        dataset2 = pd.concat([dataset2, pd.DataFrame([row])])

    # process class3
    for i in range(len(class3[0])):
        letter1, letter2, letter3 = class3[0][i]
        label = class3[1][i]

        descriptor1 = get_descriptor(letter1)
        descriptor2 = get_descriptor(letter2)
        descriptor3 = get_descriptor(letter3)

        # Interpolate descriptor so it is of fixed size
        interpolation_points = np.linspace(0, 1, N)
        descriptor1 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor1)), descriptor1)
        descriptor2 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor2)), descriptor2)
        descriptor3 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor3)), descriptor3)

        row = np.concatenate(
            (descriptor1, descriptor2, descriptor3, [label]))  # add the label and the three descriptors
        dataset3 = pd.concat([dataset3, pd.DataFrame([row])])

    return dataset1, dataset2, dataset3

