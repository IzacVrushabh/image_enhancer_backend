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
# csvName = "B"

folder='D:\Android_Projects\ExDarkImages\Car'

names_list = []
for f in glob.glob(folder+'/*.jpg'):
    names_list.append(os.path.split(f)[-1])
# print(names_list)

for f in glob.glob(folder+'/*.jpg'):
    path_list.append(f)

# print(path_list)



# Value Array
read_images = []
entropy_list = []


for image in path_list:
    read_images.append(cv.imread(image))

output_images = []
output_image_name_list = []
entropy_input = []
# Bus - BBHE, BPHEME, BHEPL,  DSIHE, MMBEBHE,
# Dog - BBHE, BPHEME, BHEPL
i = 0
for image in read_images:
    entropy_input.append(skimage.measure.shannon_entropy(image))
    ie = image_enhancement.IE(image, color_space = 'HSV')
    result = ie.BBHE()
    output_images.append(result)
    # cv.imwrite(path + "Bicycle-"+str(i)+".jpg", result)
    #
    # output_image_name_list.append(path + "Bicycle-"+str(i)+".jpg")
    # i += 1

print(output_image_name_list)

##### Entropy code...
# for image in output_images:
#     entropy_list.append(skimage.measure.shannon_entropy(image))
# print("{:.3f}".format(np.array(entropy_input).mean()))
#
# print("{:.3f}".format(np.array(entropy_list).mean()))

# # PSNR list
# psnr_list = []
# for (ip,op) in zip(read_images,output_images):
#     psnr_list.append(metrics.peak_signal_noise_ratio(ip,op))
#
#
# # brisque list
brisque_list = []
brisque_output = []
for image in path_list:
    img = PIL.Image.open(image)
    brisque_list.append(brisque.score(img))

for image in output_images:
    img = PIL.Image.fromarray(image)
    brisque_output.append(brisque.score(img))

print("{:.3f}".format(np.array(brisque_output).mean()))



# with open('BPHEME_'+ csvName +'.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["SN", "Name", "Entropy", "BRISQUE", "PSNR"])
#
#     count = 0
#     for (names, ent, brsq, psn) in zip(names_list, entropy_list, brisque_list, psnr_list):
#
#         writer.writerow([count+1, names, ent, brsq, psn])
#         count = count+1





#cv.imshow("input",input)
#cv.imshow("output",output)
cv.waitKey(0)
cv.destroyAllWindows()