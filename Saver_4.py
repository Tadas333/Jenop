import cv2
import numpy as np
import timeit
import argparse
import glob


path1 = r"C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Imige_saver/inputs/sep_4/*.mp4"
#path1 = r"C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Imige_saver/inputs/sep_4/camera9.mp4"

#use this for the frame names, so if you already did camera0.mp4 for example you would change it to 1 so it would start count from 1 not 0
start_number = 0

LABELS_FILE='C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Imige_saver/coco.names'
CONFIG_FILE='yolov3.cfg'
WEIGHTS_FILE='yolov3.weights'
CONFIDENCE_THRESHOLD=0.88

LABELS = open(LABELS_FILE).read().strip().split("\n")


for file in glob.glob(path1):
    start_number = start_number+1
    start = timeit.default_timer()
    print(start)
    if file.endswith(".mp4"):
        vidcap = cv2.VideoCapture(file)
        #success,image = vidcap.read()
        #count = 0;
        print(file)

        i=0 #frame counter
        frameTime = 1 # 
        
        while(vidcap.isOpened()):
            ret = vidcap.grab() #grab frame
            i=i+1 #increment counter
            if i % 18 == 0: # display only one third of the frames, you can change this parameter according to your needs
                ret, frame = vidcap.retrieve() 
                print(i)
                
                np.random.seed(4)
                COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                    dtype="uint8")


                net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

                #image = cv2.imread(INPUT_FILE)
                (H, W) = frame.shape[:2]

                # determine only the *output* layer names that we need from YOLO
                ln = net.getLayerNames()
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


                #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (832, 832),
                    swapRB=True, crop=False)
                net.setInput(blob)
                
                layerOutputs = net.forward(ln)
                


                #print("[INFO] YOLO took {:.6f} seconds".format(end - start))
                

                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        if classID == 8:
                            confidence = scores[classID]

                            # filter out weak predictions by ensuring the detected
                            # probability is greater than the minimum probability
                            if confidence > CONFIDENCE_THRESHOLD:
                                #cv2.imwrite("outputs/img_%3d.jpg" % count, image)
                                cv2.imwrite("outputs/img" + str(start_number) + "_%3d.jpg" % i, frame)
                                
                                #args["input"]
                                print("saved")
                    



                #cv2.imwrite("img_%3d.jpg" % count, image)   

    stop = timeit.default_timer()                      
    print('Time: ', stop - start) 
