from flask import Flask, render_template , request , jsonify
# from PIL import Image
import io, sys
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import numpy as np 
import cv2
import base64
from flask_cors import CORS

from image_enhancement import image_enhancement
from contrast_image import contrast_image
from contrast_image import quantitation
import skimage.measure
import glob
import csv
import os
import imquality.brisque as brisque
from PIL import Image
import warnings
from skimage import metrics

warnings.filterwarnings('ignore')

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = '/enhancerApp/image/'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

directory = r'D:\enhancerApp\outputImage'

@app.route('/enhance' , methods=['POST'])
def mask_image():
    p2, q2 = 0.2, 0.8
    f = request.files['image']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

    path_list = []
    names_list = []

    folder = "D:\enhancerApp\image"

    for f in glob.glob(folder + '/*.jpg'):
        names_list.append(os.path.split(f)[-1])
    # print(names_list)

    for f in glob.glob(folder + '/*.jpg'):
        path_list.append(f)
    # print(path_list)

    read_images = []
    for image in path_list:
        read_images.append(cv2.imread(image))

    for image in read_images:
        # cv2.imshow("Input-Image", image)
        ie = image_enhancement.IE(image, color_space='HSV')
        result1 = ie.BBHE()
        result2 = ie.BPHEME()

        b1, g1, r1 = cv2.split(result1)
        RESULT_image1 = (np.dstack((b1 * q2, g1 * q2, r1 * q2))).astype(np.uint8)

        b2, g2, r2 = cv2.split(result2)
        RESULT_image2 = (np.dstack((b2 * p2, g2 * p2, r2 * p2))).astype(np.uint8)

        op_image = RESULT_image1 + RESULT_image2
        output_image = cv2.cvtColor(op_image, cv2.COLOR_BGR2RGB)
        os.chdir(directory)
        cv2.imwrite("outputImageFinal.png", output_image)  # use to store expected output image in the output folder
        # # cv2.imshow("output-image", output_image)
        output_img = Image.fromarray(output_image)
        rawBytes = io.BytesIO()
        output_img.save(rawBytes, "PNG")
        rawBytes.seek(0)  # major problem....
        # response_image = base64.b64encode(output_image).decode('utf-8')
        img_base64 = base64.b64encode(rawBytes.read())

        return jsonify({'status': str(img_base64), 'ip_ent': '0', 'op_ent': '0', 'ip_brisque': '0', 'op_brisque': '0'})

    # return 'file uploaded successfully'
    # fusionValue = request.form['fusion_params']
    # p2, q2 = 0.2, 0.8
    # met1 = request.form['met1']
    # met2 = request.form['met2']
    # file = request.files['image'].read() ## byte file
    # rawBytes = io.BytesIO()
    # input_img = Image.fromarray(file)
    # input_img.save(rawBytes, format="JPG")
    # picture = input_img.save("dolls.jpg")
    #
    # return jsonify({'status': 'ok'})
    #
    #
    # npimg = np.fromstring(file, np.uint8)
    # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    # # print(npimg)
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ie = image_enhancement.IE(RGB_img, color_space='RGB')
    #
    # ip_ent = skimage.measure.shannon_entropy(RGB_img)
    # ip_brisque = brisque.score(RGB_img)
    #
    # result1 = ""
    # result2 = ""
    #
    # if met1 == 'RLBHE':
    #  result1 = ie.RLBHE()
    # elif met1 == 'DSIHE':
    #  result1 = ie.DSIHE()
    # elif met1 == 'BBHE':
    #  result1 = ie.BBHE()
    # elif met1 == 'BPHEME':
    #  result1 = ie.BPHEME()
    # elif met1 == 'FHSABP':
    #  result1 = ie.FHSABP()
    # else:
    #  result1 = ie.MMBEBHE()
    #
    # if met2 == 'RLBHE':
    #  result2 = ie.RLBHE()
    # elif met2 == 'DSIHE':
    #  result2 = ie.DSIHE()
    # elif met1 == 'BBHE':
    #  result2 = ie.BBHE()
    # elif met2 == 'BPHEME':
    #  result2 = ie.BPHEME()
    # elif met2 == 'FHSABP':
    #  result2 = ie.FHSABP()
    # else:
    #  result2 = ie.MMBEBHE()
    #
    # b1, g1, r1 = cv2.split(result1)
    # RESULT_image1 = (np.dstack((b1 * q2, g1 * q2, r1 * q2))).astype(np.uint8)
    #
    # b2, g2, r2 = cv2.split(result2)
    # RESULT_image2 = (np.dstack((b2 * p2, g2 * p2, r2 * p2))).astype(np.uint8)
    #
    # output_image = RESULT_image1 + RESULT_image2
    #
    # output_ent = skimage.measure.shannon_entropy(output_image)
    #
    # output_img = Image.fromarray(output_image)
    #
    # output_brisque = brisque.score(output_img)
    #
    # # output_psnr = metrics.peak_signal_noise_ratio(img, output_img)
    #
    # rawBytes = io.BytesIO()
    # # print(rawBytes)
    # output_img.save(rawBytes, format="PNG")
    # rawBytes.seek(0)
    # img_base64 = base64.b64encode(rawBytes.read())
    # # print(img_base64)
    # return jsonify({'status': str(img_base64), 'ip_ent': ip_ent, 'op_ent': output_ent, 'ip_brisque': ip_brisque, 'op_brisque': output_brisque})

@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)

