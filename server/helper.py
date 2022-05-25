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
import time
import shutil
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
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # print("fusion_value", request.form['fusion_params'])
    p2 = float(request.form['fusion_params'])
    q2 = 1 - float(p2)
    # print(p2, q2)
    f = request.files['image']
    met1 = request.form['met1']
    met2 = request.form['met2']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

    path_list = []
    names_list = []

    folder = "D:\enhancerApp\image"

    for f in glob.glob(folder + '/*.png'):
        names_list.append(os.path.split(f)[-1])
    # print(names_list)

    for f in glob.glob(folder + '/*.png'):
        path_list.append(f)
    # print(path_list)

    read_images = []
    for image in path_list:
        read_images.append(cv2.imread(image))

    for image in read_images:
        # cv2.imshow("Input-Image", image)
        ie = image_enhancement.IE(image, color_space='HSV')
        ip_ent = skimage.measure.shannon_entropy(image)
        ip_brisque = brisque.score(image)
        if met1 == 'RLBHE':
            result1 = ie.RLBHE()
        elif met1 == 'DSIHE':
            result1 = ie.DSIHE()
        elif met1 == 'BBHE':
            result1 = ie.BBHE()
        elif met1 == 'BPHEME':
            result1 = ie.BPHEME()
        elif met1 == 'FHSABP':
            result1 = ie.FHSABP()
        else:
            result1 = ie.MMBEBHE()

        if met2 == 'RLBHE':
            result2 = ie.RLBHE()
        elif met2 == 'DSIHE':
            result2 = ie.DSIHE()
        elif met1 == 'BBHE':
            result2 = ie.BBHE()
        elif met2 == 'BPHEME':
            result2 = ie.BPHEME()
        elif met2 == 'FHSABP':
            result2 = ie.FHSABP()
        else:
            result2 = ie.MMBEBHE()
        b1, g1, r1 = cv2.split(result1)
        result_image1 = (np.dstack((b1 * q2, g1 * q2, r1 * q2))).astype(np.uint8)

        b2, g2, r2 = cv2.split(result2)
        result_image2 = (np.dstack((b2 * p2, g2 * p2, r2 * p2))).astype(np.uint8)

        op_image = result_image1 + result_image2
        output_image = cv2.cvtColor(op_image, cv2.COLOR_BGR2RGB)
        output_ent = skimage.measure.shannon_entropy(output_image)
        output_brisque = brisque.score(output_image)
        os.chdir(directory)


        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        time_str = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite("outputImageFinal"+time_str+".jpg", output_image)  # use to store expected output image in the output folder
        # # cv2.imshow("output-image", output_image)
        output_img = Image.fromarray(output_image)
        raw_bytes = io.BytesIO()
        output_img.save(raw_bytes, "PNG")
        raw_bytes.seek(0)  # major problem....
        # response_image = base64.b64encode(output_image).decode('utf-8')
        img_base64 = base64.b64encode(raw_bytes.read())
        return jsonify({'status': str(img_base64), 'ip_ent': ip_ent, 'op_ent': output_ent, 'ip_brisque': ip_brisque, 'op_brisque': output_brisque})

    # return 'file uploaded successfully'

@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)

