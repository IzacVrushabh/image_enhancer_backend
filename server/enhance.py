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

# p = 0.2
# q = 0.8

# p1, q1 = 0.1, 0.9
p2, q2 = 0.2, 0.8
# p3, q3 = 0.3, 0.7
# p4, q4 = 0.4, 0.6


path_list = []
names_list = []

folder = 'D:\Android_Projects\Testimage'



for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)

    # param1 : DSIHE 
# print(path_list)

read_images = []
for image in path_list:
    read_images.append(cv.imread(image))

# image = cv.imread("2015_01415.jpg")
# ent2 = skimage.measure.shannon_entropy(image)
# print("input-entropy", ent2)

Avg_Entropy = []
    # cv.imshow("Input-Image", image)
    ie = image_enhancement.IE(image, color_space='HSV')
    # result1 = ie.RLBHE()
    result2 = ie.DSIHE()
    # result3 = ie.BBHE()
    result4 = ie.BPHEME()
    # result5 = ie.MMBEBHE()
    # result6 = ie.FHSABP()


    # print("MMBEBHE", skimage.measure.shannon_entropy(result1))
    # print("BPHEME", skimage.measure.shannon_entropy(result2))

    # cv.imshow("DSIHE-Result", result1)
    # cv.imshow("MMBEBHE-Result", result2)

    b1, g1, r1 = cv.split(result1)
    # DSIHE_image1 = (np.dstack((b1 * q1, g1 * q1, r1 * q1))).astype(np.uint8)
    DSIHE_image2 = (np.dstack((b1 * q2, g1 * q2, r1 * q2))).astype(np.uint8)
    # DSIHE_image3 = (np.dstack((b1 * q3, g1 * q3, r1 * q3))).astype(np.uint8)
    # DSIHE_image4 = (np.dstack((b1 * q4, g1 * q4, r1 * q4))).astype(np.uint8)

    b2, g2, r2 = cv.split(result2)
    # MMBEBHE_image1 = (np.dstack((b2 * p1, g2 * p1, r2 * p1))).astype(np.uint8)
    MMBEBHE_image2 = (np.dstack((b2 * p2, g2 * p2, r2 * p2))).astype(np.uint8)
    # MMBEBHE_image3 = (np.dstack((b2 * p3, g2 * p3, r2 * p3))).astype(np.uint8)
    # MMBEBHE_image4 = (np.dstack((b2 * p4, g2 * p4, r2 * p4))).astype(np.uint8)

    b3, g3, r3 = cv.split(result3)
    BBHE_image3 = (np.dstack((b3 * q2, g3 * q2, r3 * q2))).astype(np.uint8)
    b4, g4, r4 = cv.split(result4)
    BPHEME_image4 = (np.dstack((b4 * p2, g4 * p2, r4 * p2))).astype(np.uint8)
    b5, g5, r5 = cv.split(result5)
    MMBEBHE_image5 = (np.dstack((b5 * p2, g5 * p2, r5 * p2))).astype(np.uint8)
    b6, g6, r6 = cv.split(result6)
    FHSABP_image6 = (np.dstack((b6 * p2, g6 * p2, r6 * p2))).astype(np.uint8)


    # print("Original Entropy", ent2)
    # cv.imshow("Output", (p*result2) + (q*result1))
    # Output_Entropy_Fusion = skimage.measure.shannon_entropy(DSIHE_image1 + MMBEBHE_image1)
    # print("Output-entropy", Output_Entropy_Fusion)
    # Avg_Entropy.append(Output_Entropy_Fusion)
    # cv.imshow("Fusion-Output-DSIHE-MMBEBHE-1", DSIHE_image1 + MMBEBHE_image1)
    cv.imshow("Fusion-Output-FHSABP-DSIHE", DSIHE_image2 + FHSABP_image6)
    # cv.imshow("Fusion-Output-DSIHE-MMBEBHE-3", DSIHE_image3 + MMBEBHE_image3)
    # cv.imshow("Fusion-Output-DSIHE-MMBEBHE-4", DSIHE_image4 + MMBEBHE_image4)
    # print("Final Entropy", ent1)
    cv.imshow("Fusion-Output-MMBEBHE-DSIHE", DSIHE_image2 + MMBEBHE_image2)
    cv.imshow("Fusion-Output-BBHE-BPHEME", BBHE_image3 + BPHEME_image4)

    # print("p = ", p, ", q = ", q)

# print("Fusion-Avg-Entropy", "{:.3f}".format(np.array(Avg_Entropy).mean()))


cv.waitKey(0)
cv.destroyAllWindows()


# op = p*a + q*b
# p*a + (1-p)*b
# b + (a - b)*p  # p = 0.3
# 6.077 + (0.139)*0.3 = 6.1187