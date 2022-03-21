import cv2
import numpy as np
import glob
# import csv
import os
from image_enhancement import image_enhancement
import skimage.measure


def calcEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


if __name__ == '__main__':
    path_list = []
    folder = 'D:\Android_Projects\ExDarkImages\Entropy-images'

    names_list = []
    for f in glob.glob(folder + '/*.jpg'):
        names_list.append(os.path.split(f)[-1])
    # print(names_list)

    for f in glob.glob(folder + '/*.jpg'):
        path_list.append(f)

    read_images = []
    for image in path_list:
        read_images.append(cv2.imread(image))

    for image in read_images:
        entropy1 = calcEntropy(image)
        ie = image_enhancement.IE(image, color_space='HSV')
        result = ie.MMBEBHE()
        entropy2 = calcEntropy(result)
        print(entropy1, entropy2)
    image = cv2.imread("2015_01415.jpg", cv2.IMREAD_GRAYSCALE)
    entropy1 = calcEntropy(image)
    ie = image_enhancement.IE(image, color_space='HSV')
    result = ie.MMBEBHE()
    entropy2 = calcEntropy(image)
    print("Normal method entropy", entropy2)
    ent1 = skimage.measure.shannon_entropy(image)
    print("input-entropy", ent1)

