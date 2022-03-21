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

p_list = [0.4, 0.3, 0.2, 0.1]
q_list = [0.6, 0.7, 0.8, 0.9]


path_list = []
names_list = []

folder = 'D:\Android_Projects\ExDarkImages\Cat'

for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)
# print(path_list)

read_images = []
for image in path_list:
    read_images.append(cv.imread(image))

input_entropy_avg = 0
for image in read_images:
    Output_Entropy_Fusion = skimage.measure.shannon_entropy(image)
    input_entropy_avg += Output_Entropy_Fusion

print("Input average Entropy:", input_entropy_avg / len(read_images))

Avg_Entropy = []

Entropy_Final_list = []

para_idx = 1
for i in range(0, 4):
    idx = 0
    for image in read_images:
        ie = image_enhancement.IE(image, color_space='HSV')
        result1 = ie.BBHE()
        result2 = ie.BPHEME()

        b1, g1, r1 = cv.split(result1)
        DSIHE_image = (np.dstack((b1 * q_list[i], g1 * q_list[i], r1 * q_list[i]))).astype(np.uint8)

        b2, g2, r2 = cv.split(result2)
        MMBEBHE_image = (np.dstack((b2 * p_list[i], g2 * p_list[i], r2 * p_list[i]))).astype(np.uint8)

        Output_Entropy_Fusion = skimage.measure.shannon_entropy(DSIHE_image + MMBEBHE_image)
        Avg_Entropy.append(Output_Entropy_Fusion)
        print(para_idx, idx)
        idx = idx + 1
    para_idx = para_idx + 1

    Entropy_Final_list.append("{:.3f}".format(np.array(Avg_Entropy).mean()))

print(Entropy_Final_list)
cv.waitKey(0)
cv.destroyAllWindows()


# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187