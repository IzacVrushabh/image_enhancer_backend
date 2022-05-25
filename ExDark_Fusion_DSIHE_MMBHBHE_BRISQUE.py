from image_enhancement import image_enhancement
import cv2 as cv
from contrast_image import contrast_image
from contrast_image import quantitation
import skimage.measure
import numpy as np
import glob
import csv
import os
import imquality.brisque as brisque
import PIL.Image
import warnings
from skimage import metrics

warnings.filterwarnings('ignore')

p_list = [0.1, 0.2, 0.3, 0.4]
q_list = [0.9, 0.8, 0.7, 0.6]


path_list = []
names_list = []
old_brisque_list = []
p1_and_q1 = 0.0
p2_and_q2 = 0.0
p3_and_q3 = 0.0
p4_and_q4 = 0.0


folder = r'D:\Android_Projects\OceanDark\TempFolder'

for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)
# print(path_list)

read_images = []
for image in path_list:
    read_images.append(cv.imread(image))

# input_brisque_avg = 0
# for image in read_images:
#     input_brisque = brisque.score(image)
#     input_brisque_avg += input_brisque
#
# print("Input average Brisque:", input_brisque_avg / len(read_images))

Avg_Brisque = []

# Entropy_Final_list = []

para_idx = 1

with open('DSIHE_MMBHE_BRISQUE_For_OceanDark.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Sr No.", "Name", "Brisque", "p=0.1 & q = 0.9", "p=0.2 & q = 0.8", "p=0.3 & q = 0.7", "p=0.4 & q = 0.6"])
    file.close()

count = 0
cnt = 0
for image in read_images:
    idx = 1
    old_brisque_list.append(brisque.score(image))
    for i in range(0, 4):
        ie = image_enhancement.IE(image, color_space='HSV')
        result1 = ie.DSIHE()
        result2 = ie.MMBEBHE()

        b1, g1, r1 = cv.split(result1)
        DSIHE_image = (np.dstack((b1 * q_list[i], g1 * q_list[i], r1 * q_list[i]))).astype(np.uint8)

        b2, g2, r2 = cv.split(result2)
        MMBEBHE_image = (np.dstack((b2 * p_list[i], g2 * p_list[i], r2 * p_list[i]))).astype(np.uint8)

        img = PIL.Image.fromarray(DSIHE_image + MMBEBHE_image)
        output_brisque_fusion = brisque.score(img)

        if i == 0:
            p1_and_q1 = output_brisque_fusion
        elif i == 1:
            p2_and_q2 = output_brisque_fusion
        elif i == 2:
            p3_and_q3 = output_brisque_fusion
        else:
            p4_and_q4 = output_brisque_fusion

        print(para_idx, idx)
        idx = idx + 1
    with open('DSIHE_MMBHE_BRISQUE_For_OceanDark.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([count + 1, names_list[cnt], old_brisque_list[cnt], p1_and_q1, p2_and_q2, p3_and_q3, p4_and_q4])
        file.close()
    print("count: ", count)
    count = count + 1
    cnt = cnt + 1
    para_idx = para_idx + 1

# with open('DSIHE_MMBHE_ENTROPY'+ '.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["Sr No.", "Name", "Entropy", "p=0.1 & q = 0.9", "p=0.2 & q = 0.8", "p=0.3 & q = 0.7", "p=0.4 & q = 0.6"])
#
#             count = 0
#             for (names, old_brisque_list, p1_and_q1, p2_and_q2, p3_and_q3, p4_and_q4) in zip(names_list, old_brisque_list, p1_and_q1, p2_and_q2, p3_and_q3, p4_and_q4):
#
#                 writer.writerow([count+1, names, old_brisque_list, p1_and_q1, p2_and_q2, p3_and_q3, p4_and_q4])
#                 count = count+1

# print(Entropy_Final_list)
cv.waitKey(0)
cv.destroyAllWindows()


# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187