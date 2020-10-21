import os
import cv2
import csv
import numpy as np

from PIL import Image
from mtcnn.mtcnn import MTCNN

from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from models.util import utils

in_encoder = Normalizer('l2')
print("Using gpu: {0}".format(tf.test.is_gpu_available(
                            cuda_only=False,
                            min_cuda_compute_capability=None)))

class FaceNet:
    def __init__(self):
        self.model_Face       = self.init_model()
        self.database         = self.init_database()

    # function to read database
    def init_database(self):
        print("read data from database...")
        database = {}

        dbPath = utils.get_file_path("webApp/data", "dict.csv")
        reader = csv.reader(open(dbPath), delimiter='\n')

        for row in reader:
            data = row[0].split(",", 1)
            encode = data[1].replace('"','')
            encode = encode.replace('[','')
            encode = encode.replace(']','')
            encode = np.fromstring(encode, dtype=float, sep=',')
            database[data[0]] = encode
        return database

    def init_model(self):
        print("initial facenet model...")
        facePath = utils.get_file_path("webApp/data", "facenet_keras.h5")
       # cvPath = os.path.sep.join(["data", "haarcascade_frontalface_alt2.xml"])
        #faceCascade = cv2.CascadeClassifier(cvPath)
        return load_model(facePath, custom_objects={ 'loss': self.triplet_loss }, compile=False)

    def triplet_loss(y_true, y_pred, alpha = 0.2):
        print("using triplet_loss function...")
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss

    # get face embedding and perform face recognition
    def get_embedding(self, image):
        #print("get enbedding code function begin...")
        # scale pixel values
        face = image.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = cv2.resize(face,(160,160))
        face = np.expand_dims(face, axis=0)
        encode = self.model_Face.predict(face)[0]
        return encode

    def find_person(self, encoding, min_dist=1):
        #print("find person function begin...")
        min_dist = float("inf")
        encoding = in_encoder.transform(np.expand_dims(encoding, axis=0))[0]
        for (name, db_enc) in self.database.items():
            dist = cosine(db_enc, encoding)
            if dist < 0.5 and dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.5:
            return "None"
        else:
            return identity
        return "None"

    def recognize(self, frame, x, y, width, height):
        crop = frame[y:y+int(height), x:x+int(width)]

        #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #detect face
        #faces = faceCascade.detectMultiScale(crop,
                                            #scaleFactor=1.1,
                                            #minNeighbors=5,
                                            ##minSize=(60, 60),
                                            #flags=cv2.CASCADE_SCALE_IMAGE)
        #to draw rectangle
        #for (x, y, w, h) in faces:
        #face_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_frame = img_to_array(crop)
        name = "None"
        if face_frame.size!=0 :
            face_frame = cv2.resize(face_frame,(160, 160))
            encode = self.get_embedding(face_frame)
            name = self.find_person(encode)
        if name == "None":
            label = "Not found"
        else :
            label = name
        return label

    # use MTCNN to detect faces and return face array
    def extract_mtcnn_face(self, filename, required_size=(160, 160)):
        print("extracting face from image")
        detector = MTCNN()

        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)

        # save detected face image to allow user download
        basename = os.path.splitext(os.path.basename(filename))[0]
        outputfile = basename+"_face.jpg"
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(utils.get_file_path('webApp/static/processed', outputfile), img)

        face_array = np.asarray(image)

        return face_array

    # facenet to encode
    def extract_face(self, filename):
        image = cv2.imread(filename)
        frame = image

        # first detect face
        xmlname = 'haarcascade_frontalface_alt2.xml'
        xmlpath = utils.get_file_path('webApp/data', xmlname)
        face_detector = cv2.CascadeClassifier(xmlpath)
        faces = face_detector.detectMultiScale(frame, 1.3, 5)

        if len(faces) == 0:
            raise Exception("No face detected, please upload an image with your front face")
        elif faces.shape[0] > 1:
            print("face detected: {0}".format(faces.shape[0]))
            raise Exception ("Too many faces deteted, please upload an image with only your face")
        (x,y,w,h) = faces[0]

        # cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # generate processed image
        basename = os.path.splitext(os.path.basename(filename))[0]
        extention = os.path.splitext(os.path.basename(filename))[1]
        outputfile = basename+"_face"+extention
        croppedFrame = frame[y:y+h,x:x+w]
        cv2.imwrite(utils.get_file_path('webApp/static/processed', outputfile), croppedFrame)

        face_array = np.asarray(image)
        return face_array


    # encode person and save into db
    def save_encode_db(self, label, filename):
        print("encoding was begining for: " + label)
        imagePath = utils.get_file_path("webApp/uploads", filename)

        try:
             # extract face using facenet
            #face_frame = self.extract_face(imagePath)

             # extract face using mtcnn
            face_frame = self.extract_mtcnn_face(imagePath)

            # get enbedding code
            self.database[label] = self.get_embedding(face_frame)

            # write into db
            dbPath = utils.get_file_path("webApp/data", "dict.csv")
            with open(dbPath, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self.database.items():
                   value = list(value)
                   writer.writerow([key, value])
        except Exception as ex:
            return (400, ex.args[0])
        return (200, "face detected and encoded successfully")





