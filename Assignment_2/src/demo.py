from rotation import *
from contour import *
from descriptor import *
from dataset import *
from traintest import *
from readtext import *
import time

start_time = time.time()

x = cv2.imread("text1_v3.png")
text = "text1_v3.txt"

lines = read_text(x, text)

elapsed_time = time.time() - start_time

print("Elapsed time:", elapsed_time, "seconds")

