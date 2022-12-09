# Tadas Orentas 9/19/2022

import cv2
import glob, os
import numpy as np
from collections import Counter
import time
import random
import difflib
import datetime
import uuid
from PIL import Image
import configparser

config = configparser.RawConfigParser()

config.read_file(open(r'/home/mari1/S_001/config.txt'))

day = str(time.strftime("%m-%d-%y"))

storage = "/run/user/1000/gvfs/smb-share:server=mari_storage_2.local,share=recordings/Flir/all_outputs/"

if os.path.exists(storage + day):
    print("path exists already")
else:
    os.mkdir(storage + day)


modelConf = config.get('ALL_DET_C', 'all_det_cfg')
modelWeights = config.get('ALL_DET_C', 'all_det_weights')

checker_modelConf = config.get('ALL_DET_CHECKER', 'checker_all_det_cfg')
checker_modelWeights = config.get('ALL_DET_CHECKER', 'checker_all_det_weights')

# modelConf = '/home/mari1/S_001/all_det/yolov4-custom_fix.cfg'
# modelWeights = '/home/mari1/S_001/all_det/yolov4-custom_best_fix2.weights'

# modelConf = '/home/mari1/Desktop/switch2/all_det/yolov4-p6_custom.cfg'
# modelWeights = '/home/mari1/Desktop/switch2/all_det/yolov4-p6_custom_best.weights'

# read pre-trained model and config file
net = cv2.dnn.readNet(modelWeights, modelConf)
checker_net = cv2.dnn.readNet(checker_modelWeights, checker_modelConf)

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = None

obj = config.get('ALL_DET_C','all_det_obj')

checker_obj = config.get('ALL_DET_CHECKER', 'checker_all_det_obj')

with open(obj, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


with open(checker_obj, 'r') as f:
    checker_classes = [line.strip() for line in f.readlines()]

def checker(image2):
    def get_output_layers(checker_net):
                
        layer_names = checker_net.getLayerNames()
        
        output_layers = [layer_names[i - 1] for i in checker_net.getUnconnectedOutLayers()]

        return output_layers

    Width2 = image2.shape[1]
    Height2 = image2.shape[0]

    checker_scale = 0.00392

    checker_blob = cv2.dnn.blobFromImage(image2, checker_scale, (448,448), (0,0,0), True, crop=False)

    # set input blob for the network
    checker_net.setInput(checker_blob)
    checker_outs = checker_net.forward(get_output_layers(checker_net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.55
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in checker_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width2)
                center_y = int(detection[1] * Height2)
                w = int(detection[2] * Width2)
                h = int(detection[3] * Height2)
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
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        #draw_bounding_box(image2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        #cv2.rectangle(image2, (0,0), (60,60), 255, 4)
        
        #print("w",w,"h",h,"class",class_ids[i])
        
        if (class_ids[i] == 0) and (w>5) and (h>5): #8
            # print("ITS A BOAT")
            # print(w)
            # cv2.imwrite("all_outputs/" +str(rand)+ "_" + ".jpg", image2)
            #print("checkerid",class_ids[i])
            return(True)
#############################################################################################
def getBlurValue(image):
    canny = cv2.Canny(image, 50,250)
    return np.mean(canny)

def Base64Encode(input):
    # List of All Binary Inputs
    binary_list = [int("000000"), int("000001"), int("000010"), int("000011"), int("000100"), int("000101"), int("000110"), int("000111"), int("001000"), int("001001"), int("001010"),
                   int("001011"), int("001100"), int("001101"), int("001110"), int("001111"), int("010000"), int("010001"), int("010010"), int("010011"), int("010100"), int("010101"),
                   int("010110"), int("010111"), int("011000"), int("011101"), int("011010"), int("011011"), int("011100"), int("011101"), int("011110"), int("011111"), int("100000"),
                   int("100001"), int("100010"), int("100011"), int("100100"), int("100101"), int("100110"), int("100111"), int("101000"), int("101001"), int("101010"), int("101011"),
                   int("101100"), int("101101"), int("101110"), int("101111"), int("110000"), int("110001"), int("110010"), int("110011"), int("110100"), int("110101"), int("110110"),
                   int("110111"), int("111000"), int("111001"), int("111010"), int("111011"), int("111100"), int("111101"), int("111110"), int("111111")]
    # List of Equivalent ASCII Value
    ascii_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 
                  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']

    # Check for Valid Input
    if input in binary_list:
        # Get Index of the Input in Binary List
        index_ = binary_list.index(input)
        # Set Output to the Equivalent Index Value in ASCII List
        output = ascii_list[index_]
    elif input not in binary_list:
        print(f"ERROR: Binary Input not Found")

    return output

# Returns Site ID String
def GetSiteIDStr():
    # Set Site ID
    # Edge Computer ID Array
    # 1 = Test
    # 2 = BBI
    # 3 = HI
    # 4 = ~
    id_arr = [1, 2, 3, 4]
    id_str = str(id_arr[0])

    return id_str

# Returns Base64 Encoded UUID String
def GenerateEncodedUUID():
    # Generating UUID
    uuid_base = uuid.uuid1().int
    time.sleep(1)

    # Shortening Generated UUID to fit VRN constraints
    uuid_str = str(uuid_base)[:5]
    print(f"New UUID String Generated: {uuid_str}")

    # Converting UUID String to Binary List
    uuid_bin = ' '.join(format(x, 'b') for x in bytearray(uuid_str, 'ascii'))
    split_uuid_bin = map(int, uuid_bin.split())

    # Encoding Each Portion of the Binary UUID List
    encoded_uuid = ''
    for i in split_uuid_bin:
        temp_encoded_uuid = Base64Encode(i)
        encoded_uuid = encoded_uuid + temp_encoded_uuid

    print(f"UUID String Encoded: {encoded_uuid}")
    return encoded_uuid

# Returns Base64 Encoded Date and Time String
def GenerateEncodedDnT():
    # Set Maritime Epoch (1/1/2022 Unix Timestamp = 1640995200)
    mt_epoch_offset = 1640995200
    # Get Time Since Default Epoch
    og_epoch_time = time.time()

    # Set Date and Time String
    dnt_str = str(int(og_epoch_time - mt_epoch_offset))

    # Convert Date and Time String to Binary
    dnt_bin = str(bin(int(dnt_str))[2:])
    # Split Date and Time Binary List Every 6 Characters
    dnt_bin_list = [dnt_bin[i:i+6] for i in range(0, len(dnt_bin), 6)]

    # Get Count of Date and Time List for Padding
    dnt_bin_list_count = len(dnt_bin_list)
    if dnt_bin_list_count == 4:
        print(f"Date and Time Binary List Count = 4, Adding 2 Zeros for Spacing")
        dnt_bin_list_padding = ('0', '0')
        dnt_bin_list.extend(dnt_bin_list_padding)
    elif dnt_bin_list_count == 5:
        print(f"Date and Time Binary List Count = 5, Adding 1 Zero for Spacing")
        dnt_bin_list_padding = ('0')
        dnt_bin_list.extend(dnt_bin_list_padding)
    elif dnt_bin_list_count == 6:
        print(f"Date and Time Binary List Count = 6, No Extra Spacing Needed")

    # Base64 Encoding Date and Time Binary
    encoded_dnt = ''
    for i in dnt_bin_list:
        temp_encoded_dnt = Base64Encode(int(i))
        encoded_dnt = encoded_dnt + temp_encoded_dnt

    print(f"Date and Time String Encoded: {encoded_dnt}")
    return encoded_dnt

# Returns a Randomly Generated Base64 Encoded UID
# CALL THIS TADAS
def GenerateUID():
    # Get Site ID
    site_id = GetSiteIDStr()
    # Get Random UUID
    uuid_ = GenerateEncodedUUID()
    # Get Date and Time
    dnt_ = GenerateEncodedDnT()

    # Combine Returned Strings
    uid_ = site_id + uuid_ + dnt_

    return uid_

path = "/home/mari1/S_001/outputs/"

while(1):

    dir = os.listdir(path)
    file_count = len(dir)

    if file_count > 0:
 
        base_name = []
        match = []
        dems = []
        n3 = []
        beta =[]
        sum = []
        size = []
        chars = []

        for file in glob.glob(path + "*.jpg"):
            base_name.append(os.path.basename(file))

        for item in base_name:
            if (base_name[0][0] == item[0]) & (base_name[0][1] == item[1]) & (base_name[0][2] == item[2]) & (base_name[0][3] == item[3] )& (base_name[0][4] == item[4]):
                match.append(item)
        
        # for item in base_name:
        #     for itemm in base_name:
        #         if (item[0] == itemm[0]):
        #             match.append(item)


        for pic in match:
            try:
                image = cv2.imread(path + pic)
                Width = image.shape[1]
                Height = image.shape[0]

                area = Width*Height
                size.append(getBlurValue(image))
            except:
                print("FAILURE")
                match.clear

        if len(size)>0:
            best = max(size)
            ibest = size.index(best)
            imagecheck = cv2.imread(path + match[ibest])

            if checker(imagecheck) == True:
                image = cv2.imread(path + match[ibest])
                image2 = np.copy(image)
                Width = image.shape[1]
                Height = image.shape[0]

                channels = image.shape[2]
                scale = 0.00392

                dems.append(Width*Height*channels)

                blob = cv2.dnn.blobFromImage(image, scale, (800,800), (0,0,0), True, crop=False)
                net.setInput(blob)

                def get_output_layers(net):
                    
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    return output_layers

                def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
                    
                    label = str(classes[class_id]) +"_"+ str(round(confidence, 2))
                    beta.append(class_id)
                    color = 255, 255, 255

                    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 1)
                    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


                outs = net.forward(get_output_layers(net))

                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.3

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

                for i in indices:
                    i = i
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]

                    if class_ids[i] == 8: #8
                        #print("CHARACTERS")
                        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        im = Image.fromarray(imageRGB)
                        cimg = im.crop((round(x-20), round(y-20), round(x+w+20), round(y+h+20)))
                        chars.append(cimg)
                    
                    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

                spli3 = match[ibest].split("_")
                #print(spli3[1])

                if "L" in spli3[2]:
                    direc = "O"
                else:
                    direc = "I"

                betai = []
                for u in beta:
                    if (u not in betai) & (u in beta):
                        betai.append(u)

                sum = []
                while len(beta)>0:

                    cont = 0
                    #sum = []
                    beta2 = []
                    for cl in beta:
                        if cl == beta[0]:
                            cont += 1

                    for cc in beta:
                        if cc != beta[0]:
                            beta2.append(cc)
                    
                    beta = beta2
                    sum.append(cont)


                main = []
                for i in betai:
                    ii = betai.index(i)
                    m = classes[betai[ii]] +"_"+ str(sum[ii])
                    main.append(m)

                print(main)
                ###
                rand = int(random.uniform(000000000000, 99999999999))
                #rand = GenerateUID()
                utc_datetime = datetime.datetime.utcnow()

                tim = utc_datetime.strftime("%d-%m-%Y_%H-%M-%S")
                
                cv2.imwrite("/home/mari1/S_001/all_outputs/" +str(rand) + ".jpg", image)
                cv2.imwrite(storage + day + "/" + str(spli3[1])+ "_"+ direc + "_" + str(tim) +"_"+str(rand) + ".jpg", image2)
                
                if len(chars) > 0:
                    chars[-1].save("all_outputs/" +str(rand) +"-C" + ".jpg")
                with open("/home/mari1/S_001/all_outputs/" +str(rand) + '.txt', 'w') as f:
                    f.write(str(rand) + ".jpg" + " " + str(tim) + " " + direc + " " + str(spli3[1]) + " " + str(rand) + " " + str(main))
                    f.close()


        base_name.clear
        match.clear
        dems.clear
        n3.clear
        beta.clear
        sum.clear
        size.clear

        for filee in match:
            
            os.remove(path+filee)
                
    else:

        time.sleep(5)
        



            

            




                    