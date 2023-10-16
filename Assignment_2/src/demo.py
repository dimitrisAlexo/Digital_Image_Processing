from rotation import *
from contour import *
from descriptor import *
from dataset import *
from traintest import *
from readtext import *
import time

start_time = time.time()

x = cv2.imread("text1_v3.png")
# x = cv2.resize(x, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
text = "text1_v3.txt"

# # Find rotation angle
# angle = find_rotation_angle(x)
# print("The rotation angle is", angle, "degrees")

# c = get_contour(x)
#
# d = get_descriptor(c[0])
# print(d.shape)

lines = read_text(x, text)

elapsed_time = time.time() - start_time

print("Elapsed time:", elapsed_time, "seconds")

# img_dataset, ascii_dataset = get_dataset(x, "text2.txt")
# class1, class2, class3 = divide_into_classes(img_dataset, ascii_dataset)
# dataset1, dataset2, dataset3 = form_dataset(class1, class2, class3, 100)
#
# c1, w1 = train_test(dataset1)
# print("weighted accuracy 1: ", w1)
# c2, w2 = train_test(dataset2)
# print("weighted accuracy 2: ", w2)
# c3, w3 = train_test(dataset3)
# print("weighted accuracy 3: ", w3)
