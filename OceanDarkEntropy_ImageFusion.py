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

path_list = []
names_list = []

folder = 'D:\Android_Projects\OceanDark\TempFolder'

for f in glob.glob(folder + '/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder + '/*.jpg'):
    path_list.append(f)
# print(path_list)

read_images = []
for image in path_list:
    read_images.append(cv.imread(image))

# image = cv.imread("2015_01415.jpg")
# ent2 = skimage.measure.shannon_entropy(image)
# print("input-entropy", ent2)

Entropy_Avg_list = []

# p = 0.2
# q = 0.8
# idx = 0
#
Avg_Entropy = []
# for image in read_images:
#     # cv.imshow("Input-Image", image)
#     ie = image_enhancement.IE(image, color_space='HSV')
#     result1 = ie.BBHE()  # b
#     result2 = ie.BPHEME()  # b
#
#     b1, g1, r1 = cv.split(result1)
#     BBHE_image = (np.dstack((b1 * q, g1 * q, r1 * q))).astype(np.uint8)
#
#     b2, g2, r2 = cv.split(result2)
#     BPHEME_image = (np.dstack((b2 * p, g2 * p, r2 * p))).astype(np.uint8)
#
#     Output_Entropy_Fusion = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)
#     # print("Output-entropy", Output_Entropy_Fusion)
#     Avg_Entropy.append(Output_Entropy_Fusion)
#     idx += 1
#     print(idx)
#
# print("values : ", p, q)
# Entropy_Avg_list.append("{:.3f}".format(np.array(Avg_Entropy).mean()))
# p = 0.4
# q = 0.6
# idx = 0
#
# for image in read_images:
#     # cv.imshow("Input-Image", image)
#     ie = image_enhancement.IE(image, color_space='HSV')
#     result1 = ie.BBHE()  # b
#     result2 = ie.BPHEME()  # b
#
#     b1, g1, r1 = cv.split(result1)
#     BBHE_image = (np.dstack((b1 * q, g1 * q, r1 * q))).astype(np.uint8)
#
#     b2, g2, r2 = cv.split(result2)
#     BPHEME_image = (np.dstack((b2 * p, g2 * p, r2 * p))).astype(np.uint8)
#
#     Output_Entropy_Fusion = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)
#     # print("Output-entropy", Output_Entropy_Fusion)
#     Avg_Entropy.append(Output_Entropy_Fusion)
#     idx += 1
#     print(idx)
#
# print("values : ", p, q)
# Entropy_Avg_list.append("{:.3f}".format(np.array(Avg_Entropy).mean()))
# p = 0.3
# q = 0.7
# idx = 0
#
# for image in read_images:
#     # cv.imshow("Input-Image", image)
#     ie = image_enhancement.IE(image, color_space='HSV')
#     result1 = ie.BBHE()  # b
#     result2 = ie.BPHEME()  # b
#
#     b1, g1, r1 = cv.split(result1)
#     BBHE_image = (np.dstack((b1 * q, g1 * q, r1 * q))).astype(np.uint8)
#
#     b2, g2, r2 = cv.split(result2)
#     BPHEME_image = (np.dstack((b2 * p, g2 * p, r2 * p))).astype(np.uint8)
#
#     Output_Entropy_Fusion = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)
#     # print("Output-entropy", Output_Entropy_Fusion)
#     Avg_Entropy.append(Output_Entropy_Fusion)
#     idx += 1
#     print(idx)
#
# print("values : ", p, q)
# Entropy_Avg_list.append("{:.3f}".format(np.array(Avg_Entropy).mean()))
p = 0.1
q = 0.9
idx = 0

for image in read_images:
    # cv.imshow("Input-Image", image)
    ie = image_enhancement.IE(image, color_space='HSV')
    result1 = ie.BBHE()  # b
    result2 = ie.BPHEME()  # b

    b1, g1, r1 = cv.split(result1)
    BBHE_image = (np.dstack((b1 * q, g1 * q, r1 * q))).astype(np.uint8)

    b2, g2, r2 = cv.split(result2)
    BPHEME_image = (np.dstack((b2 * p, g2 * p, r2 * p))).astype(np.uint8)

    Output_Entropy_Fusion = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)

    # print("Output-entropy", Output_Entropy_Fusion)
    Avg_Entropy.append(Output_Entropy_Fusion)
    idx += 1
    print(idx)

print("values : ", p, q)
Entropy_Avg_list.append("{:.3f}".format(np.array(Avg_Entropy).mean()))
print(Entropy_Avg_list)

cv.waitKey(0)
cv.destroyAllWindows()

# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187