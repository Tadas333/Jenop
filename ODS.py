import cv2 as cv
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import time



#Write down conf, nms thresholds,inp width/height
confThreshold = 0.8
#confThreshold = 0.4
nmsThreshold = 0.40
#nmsThreshold = 0.50
#inpWidth = 416
#inpHeight = 416
inpWidth = 288
inpHeight = 288

cap = cv2.VideoCapture('camera99.mp4')


#Load names of classes and turn that into a list
classesFile = "obj.names"
classes = None


#Model configuration
modelConf = 'yolov4-tiny_custom.cfg'
modelWeights = 'yolov4-tiny_custom_final.weights'

#Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)

#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


#Process inputs
winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 550,416)


#cap = cv2.VideoCapture(0)
fps = FPS().start()
#start_time = time.time()
start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0



with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []
    
    

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)


def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


while cv.waitKey(1) < 0:
 #while True
   
    #get frame from video
    hasFrame, frame = cap.read()
 
	# if the frame was not grabbed, then we have reached the end
	# of the stream
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

    #cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
      
    #Create a 4D blob from a frame
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))

    postprocess (frame, outs)

    #show the image
    cv.imshow(winName, frame)
fps.stop()
