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

p = 0.2
q = 0.8


path_list = []
names_list = []

folder = 'D:\Android_Projects\OceanDark\TempFolder'



for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)
# print(path_list)

read_images = []
for image in path_list:
    read_images.append(cv.imread(image))

# image = cv.imread("2015_01415.jpg")
# ent2 = skimage.measure.shannon_entropy(image)
# print("input-entropy", ent2)

Avg_Entropy = []
for image in read_images:
    # cv.imshow("Input-Image", image)
    ie = image_enhancement.IE(image, color_space='HSV')
    result1 = ie.BBHE() #b
    result2 = ie.BPHEME() #b


    # print("MMBEBHE", skimage.measure.shannon_entropy(result1))
    # print("BPHEME", skimage.measure.shannon_entropy(result2))

    # cv.imshow("DSIHE-Result", result1)
    # cv.imshow("CLAHE-Result", result2)

    b1, g1, r1 = cv.split(result1)
    BBHE_image = (np.dstack((b1 * q, g1 * q, r1 * q))).astype(np.uint8)

    b2, g2, r2 = cv.split(result2)
    BPHEME_image = (np.dstack((b2*p, g2*p, r2*p))).astype(np.uint8)

    # cv.imshow("BPHEME-Output", BPHEME_image)


    # print("Original Entropy", ent2)
    # cv.imshow("Output", (p*result2) + (q*result1))
    Output_Entropy_Fusion = skimage.measure.shannon_entropy(BBHE_image + BPHEME_image)
    # print("Output-entropy", Output_Entropy_Fusion)
    Avg_Entropy.append(Output_Entropy_Fusion)
    # print(ent2)
    # cv.imshow("Fusion-Output-less-than-1", BBHE_image + BPHEME_image)
    # print("Final Entropy", ent1)

    # print("p = ", p, ", q = ", q)

print("Fusion-Avg-Entropy", "{:.3f}".format(np.array(Avg_Entropy).mean()))


cv.waitKey(0)
cv.destroyAllWindows()


# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187