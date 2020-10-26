import os
import threading
import argparse
import filetype
import base64
import cv2

import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

from flask import Flask, Response, make_response, send_file
from flask import flash, request, redirect, jsonify
from flask import render_template

from models.realStream import RealStream
from models.facenet import FaceNet
from models.util import utils

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")
    return render_template("index.html")

@app.route("/realstream/")
def realStream():
    # start a thread that will start a video stream
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")
    # forward to real stream page
    return render_template("realStream.html")


@app.route("/staticstream/")
def staticstream():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    # forward to static stream page
    return render_template("staticStream.html")

@app.route("/imageprocess/")
def imageprocess():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    return render_template("imageprocess.html")

@app.route("/about/")
def about():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    # forward to about page
    return render_template("about.html")

@app.route("/contact/")
def contact():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    # forward to contact page
    return render_template("contact.html")

@app.route("/imageCapture/")
def imageCapture():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    # forward to register page
    return render_template("imageCapture.html")

@app.route("/videoCapture/")
def videoCapture():
    # stop the detection thread
    global t
    try:
        t.running = False
        t.join()
    except Exception:
        print("realtime thread is not running")

    # forward to register page
    return render_template("videoCapture.html")

#---------------------------------------------------------------------
#----------------------------Functions--------------------------------
#---------------------------------------------------------------------
@app.route("/uploadfile", methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':

        # save file
        file = request.files['uploadFile']
        result = utils.save_file(file)
        if result == 0:
            print("file saved failed.")
        else:
            print("file saved successful.")
        # call function to process it
        rs = RealStream()

        # check file type
        filepath = utils.get_file_path('webApp/uploads', file.filename)
        print(filepath)
        if filetype.is_image(filepath):
            output = rs.processimage(file.filename)
        elif filetype.is_video(filepath):
            output = rs.processvideo(file.filename)
        else:
            print("delete it.")
        # allow user to download after process it
        return jsonify({'filename': output})

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the media type (mime type)
    global t

    # start a thread that will perform mask detection
    rs = RealStream()
    t = threading.Thread(target=rs.mask_detection)
    t.daemon = True
    t.start()
    return Response(rs.generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/download/<fileName>", methods=['GET'])
def download(fileName):
    file = utils.get_file_path('static/processed', fileName)

    response = make_response(send_file(file))
    response.headers["Content-Disposition"] = "attachment; filename={};".format(file)
    return response

@app.route("/content_dash", methods=['GET'])
def content_dash():
    data = request.values
    if data['type'] == 'imagecode':
        return render_template('imagecode.html')
    if data['type'] == 'imageprocess':
        return render_template('imageprocess.html')
    if data['type'] == 'folderscan':
        return render_template('folderscan.html')

@app.route('/uploadImage', methods=['GET', 'POST'])
def uploadImage():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'uploadImage' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['uploadImage']

        # save file first
        utils.save_file(file)

        # encoding and save into db
        fn = FaceNet()
        username = request.form['username']
        (status, message) = fn.save_encode_db(username, file.filename)

        response = make_response({"message":message})
        response.status_code = status
        # response.mimetype = 'text/plain'
        # response.headers['x-tag'] = 'sth.magic'
        return response


@app.route('/uploadImageBase64', methods=['GET', 'POST'])
def uploadImageBase64():
    if request.method == 'POST':
        username = request.form['username']
        imagebase64 = request.form['imageBase64']

        # convert base64 string to image
        offset = imagebase64.index(',')+1
        img_bytes = base64.b64decode(imagebase64[offset:])
        img = Image.open(BytesIO(img_bytes))
        img  = np.array(img)

        # write to file first
        filename = datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
        cv2.imwrite(utils.get_file_path('webApp/uploads', filename), utils.toRGB(img))

        # encoding and save into db
        fn = FaceNet()
        (status, message) = fn.save_encode_db(username, filename)

        # processed file name
        basename = os.path.splitext(os.path.basename(filename))[0]
        extention = os.path.splitext(os.path.basename(filename))[1]
        processedFile = basename+"_face"+extention

        response = make_response(jsonify({"message":message, "filename": processedFile}))
        response.status_code = status
        return response


# execute function
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ip address")
    ap.add_argument("-o", "--port", type=int, default=8000, help="port number of the server")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)