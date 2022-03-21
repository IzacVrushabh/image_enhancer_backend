from image_enhancement import image_enhancement
import cv2 as cv
from contrast_image import contrast_image
folderName = 'dog.jpg'
name = "dog"
input = cv.imread(folderName)

ie = image_enhancement.IE(input, color_space = 'HSV')
outputGHE = ie.GHE()
outputBBHE = ie.BBHE()
outputDSIHE = ie.DSIHE()
outputMMBEBHE = ie.MMBEBHE()
outputBPHEME = ie.BPHEME()
outputRLBHE = ie.RLBHE()
outputFHSABP = ie.FHSABP()
outputBHEPL = ie.BHEPL()


methods = ["Original", "GHE", "BBHE", "DSIHE", "MMBEBHE", "BPHEME", "RLBHE", "FHSABP", "BHEPL"]
values = [input, outputGHE, outputBBHE, outputDSIHE, outputMMBEBHE, outputBPHEME, outputRLBHE, outputFHSABP, outputBHEPL]
for i in range(9):
    cv.imshow(methods[i], values[i])
for i in range(9):
    cv.imwrite("Output-" + methods[i]+"-" + name + ".jpg", values[i])
cv.waitKey(0)
cv.destroyAllWindows()