import cv2
import numpy as np
import tensorflow as tf

from models.facenet import FaceNet
from models.util import utils


print("Using gpu: {0}".format(tf.test.is_gpu_available(cuda_only=False,
                             min_cuda_compute_capability=None)))

class MaskDetector:
    def __init__(self):
        self.facenet = FaceNet()
        self.CONFIDENCE       = 0.5
        self.THRESHOLD        = 0.3
        self.LABELS           = self.init_label()
        self.COLORS           = self.init_color()
        self.net              = self.init_weight()

    # initial labels
    def init_label(self):
        print("initial yolo label...")
        labelsPath = utils.get_file_path("webApp/cfg", "classes.names")
        return open(labelsPath).read().strip().split("\n")

    # initial colors
    def init_color(self):
        print("initial yolo colors...")
        np.random.seed(42)
        return np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

    def init_weight(self):
        print("[INFO] loading YOLO from disk...")
        # derive the paths to the YOLO weights and model configuration
        weightsPath = utils.get_file_path("webApp/data", "yolov4.weights")
        configPath = utils.get_file_path("webApp/cfg", "yolov4.cfg")
        return cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def detect(self, frame, W, H):
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # calculate execute time
        #start = time.time()
        layerOutputs = self.net.forward(ln)
        #end = time.time()

        #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        #faces_list = []
        #encodes = []
        names = []

        # def hello():
        #     print("Hello")
        # t = Timer(5.0, hello)
        # t.start()
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.CONFIDENCE:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    if classID == 0 :
                        classIDs.append(classID)
                        names.append('')
                    elif classID == 1:
                        #openCV
                        #convert to greyscale
                        #faces_list=[]
                        #encodes=[]
                        label = self.facenet.recognize(frame, x, y, width, height)

                        classIDs.append(classID)
                        names.append(label)
                        #print(label)

        # apply non-maximal suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE, self.THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}:{:.4f}".format(self.LABELS[classIDs[i]]+":"+names[i], confidences[i])
                cv2.putText(frame, text, (x, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)
