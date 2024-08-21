import cv2 as cv
import numpy as np
import cv2
import imutils
from datetime import datetime
from datetime import date
import glob, os
from scipy import spatial

#Write down conf, nms thresholds,inp width/height
confThreshold = 0.8
#confThreshold = 0.4
nmsThreshold = 0.40
#nmsThreshold = 0.50
#inpWidth = 416
#inpHeight = 416
inpWidth = 288
inpHeight = 288

hour = "00"
img_per_hour = 0

#orgin_point = (0,0)

O_pointX = 0
O_pointY = 0

det_to_O_pointX = 0
det_to_O_pointY = 0




input_dir = "C:/Users/tadas/Desktop/Yolov4/input/"
output_dir = "C:/Users/tadas/Desktop/Yolov4/output/"


#Load names of classes and turn that into a list
classesFile = "C:/Users/tadas/Desktop/Yolov4/obj.names"
classes = None


modelConf = 'C:/Users/tadas/Desktop/Yolov4/yolov4.cfg'
modelWeights = 'C:/Users/tadas/Desktop/Yolov4/yolov4.weights'

#Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)



with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    O_pointX = frameWidth/2
    O_pointY = frameHeight/2

    O_point = (O_pointX,O_pointY)

    classIDs = []
    confidences = []
    boxes = []

    detX_to_0_dist = []
    detY_to_0_dist = []

    det_to_0_dist = []
    
    

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                #if ((abs(centerX-O_pointX)<300) & (abs(centerY-O_pointY)<300)):

                # det_to_O_pointX = abs(centerX-O_pointX)
                # det_to_O_pointY = abs(centerY-O_pointY)

                # detX_to_0_dist.append(det_to_O_pointX)
                # detY_to_0_dist.append(det_to_O_pointY)

                # detX_to_0_dist.append(det_to_O_pointX)
                # detY_to_0_dist.append(det_to_O_pointY)

                # xycord = (centerX,centerY)
                # det_to_0_dist.append(xycord)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    print("indices " + str(len(indices)))
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        PcenterX = left+(width/2)
        PcenterY = top+(height/2)
        if (classIDs[i] == 2):
        
            xycord = (PcenterX,PcenterY)
            det_to_0_dist.append(xycord)

        # detX = detX_to_0_dist[i]
        # detY = detX_to_0_dist[i]
        

        #if (classIDs[i] == 2):
        
        #drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)



    tree = spatial.KDTree(det_to_0_dist)
    closest = tree.query([O_point])

    sel = int(closest[1])
    #indices[closest[1]]
    Pbox = boxes[indices[sel]]
    Pleft = Pbox[0]
    Ptop = Pbox[1]
    Pwidth = Pbox[2]
    Pheight = Pbox[3]
    drawPred(classIDs[indices[sel]], confidences[indices[sel]], Pleft, Ptop, Pleft + Pwidth, Ptop + Pheight)



def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        
    #cv.putText(img, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


#while cv.waitKey(1) < 0:
hour = "00"
while(1):
    path, dirs, files = next(os.walk(input_dir))
    #while True
    
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_hour = now.strftime("%H")
    today = date.today()
    #hour = now.strftime("%H")
    #print(current_time)
    # if(current_hour != hour):
    #     hour = current_hour

    #     print("Time" +current_time + "Images Processed" + img_per_hour)
    #     img_per_hour = 0
    
    for file in glob.glob(path + "*.JPG"):

        # read input image
        
        
        img = cv2.imread(file)
        name = os.path.basename(os.path.normpath(file))
        #print(name)

        
        blob = cv.dnn.blobFromImage(img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

        #Set the input the the net
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        postprocess (img, outs)

        #show the image
        #cv.imshow(winName, img)
        cv2.imwrite(output_dir + name, img)
        os.remove(file)
        img_per_hour += 1

    if(current_hour != hour):
        hour = current_hour

        print("Time: " + str(today) + " " + current_time + "  Images Processed: " + str(img_per_hour))
        img_per_hour = 0
#cv2.waitKey(1)
    
