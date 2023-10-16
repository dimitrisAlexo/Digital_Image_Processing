from dataset import *
from traintest import *


def read_text(img, text):
    # Trains and tests the model on the image, calculates the confusion matrix and weighted accuracy. Then reads text.
    # - img: the input image containing the text we want to read
    # - text: the text used for model training
    # - lines: list containing in each element the text (in ascii) of one line of the image

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    N = 100  # number of frequencies for each contour
    img_dataset, ascii_dataset = get_dataset(img, text, 17, 7)
    class1, class2, class3 = divide_into_classes(img_dataset, ascii_dataset)
    dataset1, dataset2, dataset3 = form_dataset(class1, class2, class3, N)

    c1, w1, knn1 = train_test(dataset1)
    print("weighted accuracy for class1: ", w1)
    c2, w2, knn2 = train_test(dataset2)
    print("weighted accuracy for class2: ", w2)
    c3, w3, knn3 = train_test(dataset3)
    print("weighted accuracy for class3: ", w3)

    xx = cv2.imread("text2_rot.png")
    # xx = cv2.resize(xx, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    img_dataset, ascii_dataset = get_dataset(xx, "text2.txt", 17, 7)

    # Predict labels for each letter and store the result in an ascii list
    lines = []
    for line_idx, line in enumerate(img_dataset):
        ascii_line = []
        for word_idx, word in enumerate(line):
            ascii_word = []
            for letter_idx, letter in enumerate(word):

                num_contours = len(img_dataset[line_idx][word_idx][letter_idx])

                if num_contours == 1:
                    descriptor = get_descriptor(letter[0])
                    interpolation_points = np.linspace(0, 1, N)
                    descriptor = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor)), descriptor)
                    row = pd.DataFrame(descriptor).T
                    predicted_label = knn1.predict(row)[0]
                elif num_contours == 2:
                    descriptor1 = get_descriptor(letter[0])
                    descriptor2 = get_descriptor(letter[1])
                    interpolation_points = np.linspace(0, 1, N)
                    descriptor1 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor1)), descriptor1)
                    descriptor2 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor2)), descriptor2)
                    row = pd.DataFrame(np.concatenate((descriptor1, descriptor2))).T
                    predicted_label = knn2.predict(row)[0]
                else:
                    descriptor1 = get_descriptor(letter[0])
                    descriptor2 = get_descriptor(letter[1])
                    descriptor3 = get_descriptor(letter[2])
                    interpolation_points = np.linspace(0, 1, N)
                    descriptor1 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor1)), descriptor1)
                    descriptor2 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor2)), descriptor2)
                    descriptor3 = np.interp(interpolation_points, np.linspace(0, 1, len(descriptor3)), descriptor3)
                    row = pd.DataFrame(np.concatenate((descriptor1, descriptor2, descriptor3))).T
                    predicted_label = knn3.predict(row)[0]

                ascii_word.append(str(predicted_label))
            ascii_line.append(ascii_word)
        lines.append(ascii_line)

    # Save the result to a text file
    with open("output.txt", "w") as f:
        for line in lines:
            for word in line:
                for letter in word:
                    f.write("".join(letter))
                f.write(" ")
            f.write("\n")

    text1 = "text2.txt"
    text2 = "output.txt"

    # Read the texts
    with open(text1, "r", encoding='utf-16') as f:
        text1_contents = f.read()
    with open(text2, "r", encoding='utf-8') as f:
        text2_contents = f.read()

    # Convert texts to lists of characters, excluding spaces and new lines
    chars1 = [c for c in text1_contents if c not in [' ', '\n']]
    chars2 = [c for c in text2_contents if c not in [' ', '\n']]

    # Calculate total number of characters and number of matching characters
    total_chars = len(chars1)
    matching_chars = sum([1 for i in range(min(len(chars1), len(chars2))) if chars1[i] == chars2[i]])
    print(matching_chars)
    print(total_chars)

    # Calculate and return the weighted accuracy as a percentage
    print((matching_chars / total_chars) * 100)

    return lines
