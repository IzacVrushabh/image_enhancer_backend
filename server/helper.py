from flask import Flask, render_template , request , jsonify
# from PIL import Image
import os , io , sys
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

app = Flask(__name__)
CORS(app)

############################################## THE REAL DEAL ###############################################
@app.route('/enhance' , methods=['POST'])
def mask_image():
    fusionValue = request.form['fusion_params']
    fusionValue = int(float(fusionValue))
    p2, q2 = 0.2, 0.8
    met1 = request.form['met1']
    met2= request.form['met2']
    file = request.files['image'].read() ## byte file
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    # print(npimg)
	# ######### Do preprocessing here ################
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ie = image_enhancement.IE(RGB_img, color_space='RGB')

    ip_ent = skimage.measure.shannon_entropy(RGB_img)
    ip_brisque = brisque.score(RGB_img)

    result1=""
    result2=""

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
    else : 
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
    else :
     result2 = ie.MMBEBHE()

    b1, g1, r1 = cv2.split(result1)
    RESULT_image1 = (np.dstack((b1 * q2, g1 * q2, r1 * q2))).astype(np.uint8)

    b2, g2, r2 = cv2.split(result2)
    RESULT_image2 = (np.dstack((b2 * p2, g2 * p2, r2 * p2))).astype(np.uint8)

	# ################################################
    output_image = RESULT_image1 + RESULT_image2

    output_ent = skimage.measure.shannon_entropy(output_image)

    output_img = Image.fromarray(output_image)

    output_brisque = brisque.score(output_img)

    # output_psnr = metrics.peak_signal_noise_ratio(img, output_img)

    rawBytes = io.BytesIO()
    output_img.save(rawBytes, "PNG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64),'ip_ent': ip_ent,'op_ent': output_ent, 'ip_brisque': ip_brisque, 'op_brisque': output_brisque})
	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)

