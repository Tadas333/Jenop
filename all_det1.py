# import required packages
import cv2
import glob, os
import numpy as np
from pathlib import Path

# handle command line arguments
#c = "C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/all_detector/obj"
pathlist = Path("C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/boat_only/outputs").rglob('*.jpg')
path = "C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/boat_only/outputs/"

counter = 1

#for file in glob.glob("*.JPG"):

while(1):
    #for path in pathlist:
    for file in glob.glob(path + "*.JPG"):
        # read input image
        image = cv2.imread(file)
        count = 0
        # for n in file:
        #     if n == '/':
        #         count = count + 1
        #     if count  % 7 == 0:
        #         #a,name = file.rpartition('/')[0]
        #         a,name = file.partition('/')
        #print(name)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # read class names from text file
        classes = None
        with open('obj.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
       
    
        #print(classes)

        modelConf = 'yolov4-custom.cfg'
        modelWeights = 'yolov4-custom_best.weights'


        # read pre-trained model and config file
        net = cv2.dnn.readNet(modelWeights, modelConf)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (576,576), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # function to get the output layer names 
        # in the architecture
        def get_output_layers(net):
            
            layer_names = net.getLayerNames()
            
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers

        # function to draw bounding box on the detected object with class name
        def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            
            label = str(classes[class_id])
            # if(class_id == 1):
            #     color = 100
            # if(class_id == 2):
            #     color = 150     
            # if(class_id == 3):
            #     color = 200
            # if(class_id == 4):
            #     color = 200         
            #print(class_id)
            
            for i in range(class_id):
                color = i*9.4, i*9.4, i*9.4

            if class_id == 3:
                color = 0, 0, 0
            if class_id == 4:
                color = 100, 0, 0
            if class_id == 5:
                color = 0, 100, 0
            if class_id == 6:
                color = 0, 0, 100
            if class_id == 6:
                color = 0, 0, 100
            if class_id == 7:
                color = 100, 0, 100
            if class_id == 8:
                color = 100, 100, 100
            if class_id == 9:
                color = 200, 100, 0
            if class_id == 10:
                color = 0, 100, 200
            if class_id == 11:
                color = 200, 100, 200

            #color = 255, 255, 255

            #color = COLORS[class_id]

            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        # display output image    
        #cv2.imshow("object detection", image)

        # wait until any key is pressed
        #cv2.waitKey()
            
        # save output image to disk
        cv2.imwrite("all_outputs/" + str(counter) + ".jpg", image)
        counter = counter + 1
        print(counter)
        # release resources
        #cv2.destroyAllWindows()
        cv2.waitKey()
    if cv2.waitKey():
        break