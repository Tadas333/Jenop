import cv2 as cv
#from matplotlib.style import available
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import time
from test7 import CentroidTracker
import math
from statistics import mean
from PIL import Image
from numpy import interp
from math import sqrt
import glob, os
import sys


os.chdir('C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/switch_1')


#Write down conf, nms thresholds,inp width/height
confThreshold = 0.5
#confThreshold = 0.4
nmsThreshold = 0.30
#nmsThreshold = 0.50
# inpWidth = 448
# inpHeight = 448
inpWidth = 416
inpHeight = 416
# inpWidth = 288
# inpHeight = 288

#cap = cv2.VideoCapture('test_videos/camera104.mp4')
cap = cv2.VideoCapture('test_videos/thermal112.mp4')
#cap = cv2.VideoCapture("rtspsrc location=rtsp://admin:Jenoptik33478@192.168.1.220/h264/stream3 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)


#thermal100
# left_border = 30
# right_border = 630

#camera103
left_border = 140
right_border = 770

#camera102
# left_border = 400
# right_border = 3600

#camera100
# left_border = 100
# right_border = 2900

#camera101.mp4
# left_border = 2000
# right_border = 3600
#camera100L
# left_border = 3840 - 2600
# right_border = 3840 - 400

ocean = str("left")

start_count = 0


swap1 = 0
swap2 = 0

last_v1 = 0
last_v2 = 0

lol1 = []
lol2 = []

liner1 = []
liner2 = []

liner11 = []
liner22 = []

slop_d1 = []
slop_d2 = []

imc1 = []
imc2 = []

cords = []
cords1 = []
cords2 = []

countX = []
countY= []

images1 = []
images2 = []

images10 = []
images20 = []

Rw = []
Lw = []

RRw = []
LLw = []

test1 = []
test2 = []
wipe = []

nem = [0]

#Load names of classes and turn that into a list
classesFile = "obj.names"
classes = None


#Model configuration
modelConf = 'yolov4-tiny_custom.cfg'
#modelWeights = 'yolov4-tiny_custom_final.weights'
modelWeights = 'yolov4-tiny_custom_final_thermal.weights'

#Set up the net

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)

# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


#Process inputs
winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 650,516)

frame_rate = 14
prev = 0

fps = FPS().start()
start_time = time.time()
start_timee = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

jstart = time.strftime("%d-%m-%Y_%H-%M-%S")

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                ###MAIN TRACKER DIST VALUE
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

tracker = EuclideanDistTracker()


with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')






def do():
    if len(cords1)>1:
        pp_result = slope_integrator(cords1)
        pp = pp_result[0]
        result = pp_result[1]
        lim_result = round(result, 2)
        
        timeleft = lim_result/len(liner1)
        
        endcord = pp[1]
        remain_dist = right_border-endcord[0]

        slop_d1.append(remain_dist)    
        liner11.append(timeleft)
        test1.append(pp)
        imc1.append(len(liner1))
        lol1.append(iei[-1])
        images10.append(images1.copy())
        RRw.append(Rw.copy())
        

        # images1.clear()
        # Rw.clear()
        

        

    if len(cords2)>1:
        pp_result = slope_integrator(cords2)
        pp = pp_result[0]
        result = pp_result[1]
        lim_result = round(result, 2)

        timeleft1 = lim_result/len(liner2)
        
        endcord1 = pp[0]
        remain_dist1 = endcord1[0] - left_border

        slop_d2.append(remain_dist1)    
        liner22.append(timeleft1)
        test2.append(pp)
        lol2.append(iei[-1])
        imc2.append(len(liner2))
        images20.append(images2.copy())
        LLw.append(Lw.copy())
        

        # images2.clear()
        # Lw.clear()
        
        #print("1_1",len(images20[-1]),len(LLw[-1]))


    cords.clear()
    cords1.clear()
    cords2.clear()
    countX.clear()
    countY.clear()
    liner1.clear()
    liner2.clear()
    images2.clear()
    Lw.clear()
    images1.clear()
    Rw.clear()

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))  
    b = mean(ys) - m*mean(xs)
    
    return m, b

def slope_integrator(cor):
    midd = int(len(cor)/2)
    mid = cor[midd]

    last = cor[-1] 
    first = cor[0]

    xs = np.array([first[0],mid[0],last[0]], dtype=np.float64)
    ys = np.array([first[1],mid[1],last[1]], dtype=np.float64)

    m, b = best_fit_slope_and_intercept(xs,ys)

    regression_line = [(m*x)+b for x in xs]
    regression_line = []
    for x in xs:
        regression_line.append((m*x)+b)
    
    xp= int(xs[0]) 
    xpp= int(xs[-1])
    sp= int(regression_line[0])
    spp= int(regression_line[-1]) 

    d1= int(first[0]) 
    d2= int(first[1])
    d3= int(last[0])
    d4= int(last[1]) 

    result = ((((d4 - d2 )**2) + ((d3-d1)**2) )**0.5)
    pp = (xp,sp),(xpp,spp)

    return(pp,result)
    



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
            
                

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

   
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

    
        if( left > left_border & left+width < right_border):

            countX.append(centerX)
            countY.append(centerY)

            area = left*height
           

            


            imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            #im = Image.fromarray(imageRGB)
            im1 = Image.fromarray(imageRGB)
            #im1 = im.crop((left-100, top-100, left+width+100, top+height+100))
            #im1 = im.crop((left-30, top-30, left+width+30, top+height+30))
            

            label = '%.2f' % confidences[i]
            confy = '%.2f' % (confidences[i] * 100)
            #width = '%.2f' % width
            #ww = '%.2f' % (width)
            #ll = '%.2f' % (height)

            h2 = int(centerY - height/2)
            # percy = (h2 * 100)/2160
            # percx = (centerX * 100)/3840
            width = int(width)

            #sh = interp(h2,[0,2160],[110,35])
            sh = interp(h2,[140,570],[20,30])
            #sw = interp(width,[0,3840],[110,45])
            sw = interp(width,[0,910],[10,30])
            hypo = sqrt(sw**2 + sh**2)
            ww = int((width/hypo)*3)
            

         
            #if (len(countX)>3) & (area > 150000):
            if (len(countX)>3):
                #if(abs(countX[-1]-countX[-2])<200) & (abs(countY[-1]-countY[-2])<100):
                if(abs(countX[-1]-countX[-2])<100) & (abs(countY[-1]-countY[-2])<100) & (abs(countY[-2]-countY[-3])<100):
                    #if(((abs(countX[-1]-right_border)>300)) & (abs(countX[-1]-left_border)>300)):
                    #if((countX[-1]<right_border) & (countX[-1]>left_border)):
                    if(((left+width)<right_border) & (left>left_border)):
                        #print("rght", left+width, right_border,"left",left , left_border)
                        cords.append((centerX,centerY))

                        #if( countX[-1]>countX[-2]>countX[-3]>countX[-4]):
                        if( countX[-1]>countX[-2]>countX[-3]):
                            cords1.append((centerX,centerY))
                            images1.append(im1)
                            liner1.append('1')
                            Rw.append(ww)
                            
       
                        #if( countX[-1]<countX[-2]<countX[-3]<countX[-4]):
                        if( countX[-1]<countX[-2]<countX[-3]):
                            cords2.append((centerX,centerY))
                            images2.append(im1)
                            liner2.append('1')
                            Lw.append(ww)
                            

                        if((countX[-1]>countX[-2]>countX[-3]) & (abs((left+width)-right_border)<100)) | ((countX[-1]<countX[-2]<countX[-3]) & (abs(left-left_border)<100)):
                            do()
                            #print("OOOright", left+width, right_border,"left",left, left_border)

                else:
                    do()
 

            box1 = box * np.array([2, 1.7, 2, 1.7])
            #box1 = box * np.array([1])
            
            rects.append(box1.astype("int"))

            boxes.append([left, top, width, height])

            if classes:
                assert (classIDs[i] < len(classes))
                label = '%s:%s' % (classes[classIDs[i]], label)
            
            text = (str(label))
            boxes_ids = tracker.update(boxes)
            eklenecekler = []
            filtered =[]

            for i in range(len(boxes_ids)):
                if boxes_ids[i][4] not in eklenecekler:
                    eklenecekler.append(boxes_ids[i][4])
                    filtered.append(boxes_ids[i])
        
            font = cv.FONT_HERSHEY_SIMPLEX

            
            for box_id in filtered:
                x, y, w, h, id = box_id
                cv.putText(frame, str(confy), (left , top), font, 1, (255,255,255), 1) # count yazısı
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)

    
    
    if len(indices) == 0:
        #print(len(wipe))
        wipe.append('1')
        if len(wipe)>10:
                #print('now')
                do()
                wipe.clear()



def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

i = 0
iei = [0]
while cv.waitKey(1) < 0:
    time_elapsed = time.time() - prev
    #ret = cap.grab() #grab frame
    i=i+1 #increment counter
    iei = [i]

    ret, frame = cap.read()

    
    
    #if i % 2 == 0: # display only one third of the frames, you can change this parameter according to your needs
            #ret, frame = cap.retrieve()
    if time_elapsed > 1./frame_rate:
            #prev = time.time()

            # start_count = time.time()

            # if start_count > (start_timee + 2):
            #     switch(frame)
            #     start_timee = time.time() 


            # if time_elapsed > 1./frame_rate:
            #     prev = time.time()

  

            rects = []
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            width  = cap.get(3)  # float `width`
            height = cap.get(4)

            #print(width,height)

            counter+=1
            if (time.time() - start_time) > x :
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

            #cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            fps.update()
            
            #Create a 4D blob from a frame
            #frame = ((frame / np.max(frame)) * 255).astype('uint8')
            
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

            #Set the input the the net
            net.setInput(blob)
            outs = net.forward (getOutputsNames(net))

            postprocess (frame, outs)

            if len(countX)>50:
                countX = countX[50:]
            if len(countY)>50:
                countY = countY[50:]

            

            
            for ii, element in enumerate(cords): 
                p = cords[ii-1] if ii > 0 else 0
                c = element 
                n = cords[ii+1] if ii < len(cords)-1 else 0 
                if (c == 0) | (n == 0):
                    break
                
                cv2.line(frame, c,n, (0, 255, 255), thickness=1, lineType=1)


            for item in test1:
                cv2.line(frame, item[0],item[1], (220, 50, 255), thickness=1, lineType=1)
                for iii, element in enumerate(test1): 
                    pp = test1[iii-1] if iii > 0 else 0
                    cc = element 
                    nn = test1[iii+1] if iii < len(test1)-1 else 0 

                    if (cc == 0) | (nn == 0):
                        break
                    
                    ccc = cc[1]
                    nnn = nn[0]

                    if (ccc[0]<nnn[0]):
                        cv2.line(frame, ccc,nnn, (220, 50, 255), thickness=1, lineType=1)
                        
                
                if(len(test1)>1):
                    if test1[-1] != last_v1:
                        mam1 = test1[-1]
                        mam2 = test1[-2]
                        mamm3 = mam1[0]
                        mamm4 = mam2[1]
                        if mamm3[0]>mamm4[0]:   
                            last_v1 = test1[-1]
                            samp1 = sum(imc1)
                            #samp1 = imc1[-2] + imc1[-1]
                            samp11 = []
                            wamp11 = []
                            hamp11 = []
                            for x in range(len(images10)):
                                samp11 += images10[x]
                            for y in range(len(RRw)):
                                wamp11 += RRw[y]
 
                            #print("1",len(images10[-1]))
                            #samp11 = images10[-2] + images10[-1]
                            imc1.pop()
                            images10.pop()
                            RRw.pop()
                            
                            imc1.append(samp1)
                            images10.append(samp11)
                            RRw.append(wamp11)
                           
                            #print("2",len(images10[-1]))
                            


                if(len(liner11)>0) & (len(slop_d1)>0) & (len(lol1)>0):
                    
                    ite2 = item[1] 

                    div = sum(liner11)/len(liner11)
                    if div<10:
                        div = 10
                    if div>100:
                        div = 100
                    if ocean == "left":
                        
                        #catcher = (slop_d1[-1]/liner11[-1]) + ((slop_d1[-1]/liner11[-1])/0.8)
                        catcher = (sum(slop_d1)/len(slop_d1))/(div)
                    if ocean == "right":
                        catcher = (slop_d1[-1]/liner11[-1]) + ((slop_d1[-1]/liner11[-1])/0.4)

           

                    #print("1timer",iei[-1]-lol1[-1],"catcher2",catcher,"ite2[0]",ite2[0],"slope",len(slop_d1),sum(slop_d1)/len(slop_d1),"div",len(liner11),div)
                    #if not abs(lastcatcher1-catcher)<100:
                    #if ((iei[-1]-lol1[-1])>catcher) | (abs(ite2[0]-right_border)<100) | (int(catcher)>1000): 
                    if ((iei[-1]-lol1[-1])>catcher) | (abs(ite2[0]-right_border)<100): 
                    #if ((iei[-1]-lol1[-1])>catcher): 
                    
                        if ocean == "left":
                            imagemid = widR = int(imc1[-1]*0.85) 
                            imagemid_2 = widR_2 = int(imagemid*0.8)
                            imagemid_3 = widR_3 = int(imagemid*0.6)
                            
                                
                            if((imc1[-1] < 6) | (swap1 == 1)):
                                imagemid = widR = int(len(images10[-1])/1.5)
                                imagemid_2 = widR_2 = int(imagemid*0.9)
                                imagemid_3 = widR_3 = int(imagemid*1.1)
                                

                            # if(imagemid >= len(images10[-1])):
                            #     imagemid = widR = int(len(images10[-1])/2)
                            #     imagemid_2 = widR_2 = int(imagemid*0.8)
                            #     imagemid_3 = widR_3 = int(imagemid*0.6)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                                


                        if ocean == "right":
                            imagemid = widR = int(imc1[-1]*0.15)
                            imagemid_2 = widR_2 = int(imagemid*1.2)
                            imagemid_3 = widR_3 = int(imagemid*1.5)
                                
                            if((imc1[-1] < 6) | (swap1 == 1)):
                                imagemid = widR = int(len(images10[-1])/1.5)
                                imagemid_2 = widR_2 = int(imagemid*1.1)
                                imagemid_3 = widR_3 = int(imagemid*0.9)
                            
                            # if(imagemid >= len(images10[-1])):
                            #     imagemid = widR = int(len(images10[-1])/2)
                            #     imagemid_2 = widR_2 = int(imagemid*1.2)
                            #     imagemid_3 = widR_3 = int(imagemid*1.5)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                        
                            

                        print("R1_lenim1:",len(images10[-1]),"imageid1",imagemid,"lentest1:",len(test1),"width",RRw[-1][widR] ,"liner11:",len(liner11),"slop1:",len(slop_d1),"imc1:",len(imc1), imc1[-1],"nem:",nem[-1],"testl",len(test1))

                        

                        # images10[-1][imagemid].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TR1" + "_" + str(RRw[-1][widR]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images10[-1][imagemid_2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TR2" + "_" + str(RRw[-1][widR_2]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images10[-1][imagemid_3].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TR3" + "_" + str(RRw[-1][widR_3]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        #

                        images10[-1][imagemid].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-1][widR]) + "_" + str(nem[-1]) + "R1.jpg")
                        images10[-1][imagemid_2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-1][widR_2]) + "_" + str(nem[-1]) + "R2.jpg")
                        images10[-1][imagemid_3].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-1][widR_3]) + "_" + str(nem[-1]) + "R3.jpg")


                        # images10[-1][imagemid].save("outputs/" + str(nem[-1]) + "_" + "R1" + "_" + str(RRw[-1][widR]) + "_" + str(len(images10[-1])) + "-" + str(imagemid) + ".jpg")
                        # images10[-1][imagemid_2].save("outputs/" + str(nem[-1]) + "_" + "R2" + "_" + str(RRw[-1][widR_2]) + "_" + str(len(images10[-1])) + "-" + str(imagemid_2) + ".jpg")
                        # images10[-1][imagemid_3].save("outputs/" + str(nem[-1]) + "_" + "R3" + "_" + str(RRw[-1][widR_3]) + "_" + str(len(images10[-1])) + "-" + str(imagemid_3) + ".jpg")
                        # images10[-1][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "right2" + "_" + str(len(images10[-1])) + "-" + str(imagemid2) + ".jpg")
                        #images10[-1][imagemid3].save("outputs/" + str(nem[-1]) + "_" + "right3" + "_" + str(len(images10[-1])) + "-" + str(imagemid3) + ".jpg")
                        nem[0]+=1
                        lennn = int(len(test1)-1)

                        if(len(test1)>1):
                            dip1 = test1[-1]
                            dip2 = test1[-2]
                            dip3 = dip1[0]
                            dip4 = dip2[0]
                            #print(dip3[0],dip4[0])
                            if (dip3[0]>dip4[0]):
                            #if (dip3[0]>dip4[0]) | (dip3[0]<dip4[0]):
                                #print("HHHHHHHHHHH")
                                images10.clear()
                                test1.clear()
                                slop_d1.clear()
                                liner11.clear()
                                lol1.clear()
                                imc1.clear()
                                RRw.clear()
                                
                                
                                #print('full',len(images1))
    
                        else:
                            images10 = images10[:lennn]
                            test1 = test1[:lennn]
                            liner11 = liner11[:lennn]
                            slop_d1 = slop_d1[:lennn]
                            lol1 = lol1[:lennn]
                            imc1 = imc1[:lennn]
                            RRw = RRw[:lennn]
                            
                        swap1 = 0
                        
                
                if len(test1)>1:
                    ite3 = test1[-2]
                    ite4 = test1[-1]
                    ite5 = ite4[0]
                    ite6 = ite3[0]
                    sert1 = ite3[1]
                    sert2 = ite4[1]

                    sert11 = (ite6[0]+sert1[0])/2
                    sert22 = (ite5[0]+sert2[0])/2
                    
                    #print("liner1[-1]:",(liner11[-1]),"Lentest1:",len(test1), int(liner11[-1])/len(test1))
                    #print(test1)
                    #print("ser11",sert11,"ser11",sert11,"ite6[1]",ite6[1],"ite5[1]",ite5[1],"ite6[0]",ite6[0],"ite5[0]",ite5[0])
                    if(sert11>sert22) | (abs(ite6[1]-ite5[1])>250) | (ite6[0]>ite5[0]):
                        
                    #if(sert11>sert22):
                        if ocean == "left":
                            imagemid = widR = int(imc1[-2]*0.85) 
                            imagemid_2 = widR_2 = int(imagemid*0.8)
                            imagemid_3 = widR_3 = int(imagemid*0.6)
                            
                                
                            if((imc1[-2] < 6) | (swap1 == 1)):
                                imagemid = widR = int(len(images10[-2])/1.5)
                                imagemid_2 = widR_2 = int(imagemid*0.9)
                                imagemid_3 = widR_3 = int(imagemid*1.1)
                                

                            # if(imagemid >= len(images10[-2])):
                            #     imagemid = widR = int(len(images10[-2])/2)
                            #     imagemid_2 = widR_2 = int(imagemid*0.8)
                            #     imagemid_3 = widR_3 = int(imagemid*0.6)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                            


                        if ocean == "right":
                            imagemid = widR = int(imc1[-2]*0.15)
                            imagemid_2 = widR_2 = int(imagemid*1.2)
                            imagemid_3 = widR_3 = int(imagemid*1.5)
                                
                            if((imc1[-2] < 6) | (swap1 == 1)):
                                imagemid = widR = int(len(images10[-2])/1.5)
                                imagemid_2 = widR_2 = int(imagemid*1.1)
                                imagemid_3 = widR_3 = int(imagemid*0.9)
                            
                            # if(imagemid >= len(images10[-2])):
                            #     imagemid = widR = int(len(images10[-2])/2)
                            #     imagemid_2 = widR_2 = int(imagemid*1.2)
                            #     imagemid_3 = widR_3 = int(imagemid*1.5)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

                        print("R2_lenim11:",len(images10[-1]),"imageid1",imagemid,"lentest1:",len(test1),"width",RRw[-2][widR] ,"liner11:",len(liner11),"slop1:",len(slop_d1),"imc1:",len(imc1), imc1[-1],"nem:",nem[-1])
                        
                        # images10[-2][imagemid].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TRR1" + "_" + str(RRw[-2][widR]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images10[-2][imagemid_2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TRR2" + "_" + str(RRw[-2][widR_2]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images10[-2][imagemid_3].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TRR3" + "_" + str(RRw[-2][widR_3]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # #

                        images10[-2][imagemid].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-2][widR]) + "_" + str(nem[-1]) + "RR1.jpg")
                        images10[-2][imagemid_2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-2][widR_2]) + "_" + str(nem[-1]) + "RR2.jpg")
                        images10[-2][imagemid_3].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(RRw[-2][widR_3]) + "_" + str(nem[-1]) + "RR3.jpg")


                        # images10[-2][imagemid].save("outputs/" + str(nem[-1]) + "_" + "RR1" + "_" + str(RRw[-2][widR]) + "_" + str(len(images10[-2])) + "-" + str(imagemid) + ".jpg")
                        # images10[-2][imagemid_2].save("outputs/" + str(nem[-1]) + "_" + "RR2" + "_" + str(RRw[-2][widR_2]) + "_" + str(len(images10[-2])) + "-" + str(imagemid_2) + ".jpg")
                        # images10[-2][imagemid_3].save("outputs/" + str(nem[-1]) + "_" + "RR3" + "_" + str(RRw[-2][widR_3]) + "_" + str(len(images10[-2])) + "-" + str(imagemid_3) + ".jpg")
                        # images10[-1][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "right12" + "_" + str(len(images10[-1])) + "-" + str(imagemid2) + ".jpg")
                        #images10[-1][imagemid3].save("outputs/" + str(nem[-1]) + "_" + "right13" + "_" + str(len(images10[-1])) + "-" + str(imagemid3) + ".jpg")
                        nem[0]+=1

                        lennn = int(len(test1)-1)  
                               

                        # images10.pop(-2)   
                        # test1.pop(-2)   
                        # liner11.pop(-2)   
                        # slop_d1.pop(-2)   
                        # #lol1 = lol1[leennn:]
                        # lol1.pop(-2)   
                        # imc1.pop(-2)   
                        # RRw.pop(-2)   

                        images10 = images10[lennn:]
                        test1 = test1[lennn:]
                        liner11 = liner11[lennn:]
                        slop_d1 = slop_d1[lennn:]
                        #lol1 = lol1[leennn:]
                        #lol1 = lol1[lennn:]
                        lol1 = lol1[lennn:]
                        imc1 = imc1[lennn:]
                        RRw = RRw[lennn:]
                        
                        swap1 = 1

                    
                            


            for itemm in test2:
                cv2.line(frame, itemm[0],itemm[1], (255, 128, 128), thickness=1, lineType=1) 
                for iiii, element in enumerate(test2): 
                    ppp = test2[iiii-1] if iiii > 0 else 0
                    ccc = element 
                    nnn = test2[iiii+1] if iiii < len(test2)-1 else 0 

                    if (ccc == 0) | (nnn == 0):
                        break
                    
                    cccc = ccc[1]
                    nnnn = nnn[0]

                    if (cccc[0]>nnnn[0]):
                        cv2.line(frame, cccc,nnnn, (255, 128, 128), thickness=1, lineType=1)

                if(len(test2)>1):
                    if test2[-1] != last_v2:
                        #print(test2)
                        maam1 = test2[-1]
                        maam2 = test2[-2]
                        maamm3 = maam1[0]
                        maamm4 = maam2[1]
                        #print("imc2",imc2,"maamm3",maamm3[0],"maamm4",maamm4[0])
                        if maamm3[0]<maamm4[0]:   
                            #print("idk tbh111",len(images20[-1]),len(LLw[-1]))
                            last_v2 = test2[-1]
                            samp2 = sum(imc2)
                            #samp2 = imc2[-2] + imc2[-1]
                            #print("11",len(images20[-1]))
                            #samp22 = images20[-2] + images20[-1]
                            samp22 = []
                            wamp22 = []
                            hamp22 = []
                            for x in range(len(images20)):
                                samp22 += images20[x]
                            for y in range(len(LLw)):
                                wamp22 += LLw[y]
     
                            imc2.pop()
                            images20.pop()
                            LLw.pop()
                            
                            imc2.append(samp2)
                            images20.append(samp22)
                            LLw.append(wamp22)
                            
                            #print("idk tbh222",len(images20[-1]),len(LLw[-1]))
                            #print("22",len(images20[-1]))

                            
                
                if(len(liner22)>0) & (len(slop_d2)>0) & (len(lol2)>0):
                    
                    itee2 = itemm[1]

                    div2 = sum(liner22)/len(liner22)
                    if div2<10:
                        div2 = 10
                    if div2>100:
                        div2 = 100
                    if ocean == "left":
                        #catcher2 = (slop_d2[-1]/liner22[-1]) + ((slop_d2[-1]/liner22[-1])/0.4)
                        catcher2 = (sum(slop_d2)/len(slop_d2))/(div2)
                    if ocean == "right":
                        catcher2 = (slop_d2[-1]/liner22[-1]) + ((slop_d2[-1]/liner22[-1])/0.8)#0.8
                    #catcher2 = ((slop_d2[-1]/liner22[-1]) + 50)
                    catcher2 = int(catcher2)


                    #print("                                                              2timer",iei[-1]-lol2[-1],"catcher2",catcher2,"itee2[0]",itee2[0],"slop_d2[-1]",len(slop_d2),sum(slop_d2)/len(slop_d2),"div2",div2)
                      
                    #if (((iei[-1]-lol2[-1])>catcher2) | (abs(itee2[0]-left_border)<100) | (int(catcher2)>1000)):
                    if ((iei[-1]-lol2[-1])>catcher2) | (abs(itee2[0]-left_border)<200):
                        #print("idk tbh",len(images20[-1]),len(LLw[-1]))
                        if ocean == "left":
                            imagemid2 = widL = int(imc2[-1]*0.15)
                            imagemid2_2 = widL_2 = int(imagemid2*1.2)
                            imagemid3_3 = widL_3 = int(imagemid2*1.5)
                                                                            
                            if((imc2[-1] < 6) | (swap2 == 1)):
                                imagemid2 = widL = int(len(images20[-1])/1.5)
                                imagemid2_2 = widL_2 = int(imagemid2*1.1)
                                imagemid3_3 = widL_3  = int(imagemid2*0.9)           
                            
                            # if(imagemid2 >= len(images20[-1])):
                            #     imagemid2 = widL = int(len(images20[-1])/2)
                            #     imagemid2_2 = widL_2 = int(imagemid2*1.2)
                            #     imagemid3_3 = widL_3 = int(imagemid2*1.5)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                                
                                                  

                        if ocean == "right":
                            imagemid2 = widL = int(imc2[-1]*0.85)
                            imagemid2_2 = widL_2 = int(imagemid2*0.8)
                            imagemid3_3 = widL_3 = hiL_3 = int(imagemid2*0.6)
                    
                            if((imc2[-1] < 6) | (swap2 == 1)):
                                imagemid2 = widL = int(len(images20[-1])/1.5)
                                imagemid2_2 = widL_2 = int(imagemid2*0.9)
                                imagemid3_3 = widL_3 = int(imagemid2*1.1)
                            
                            # if(imagemid2 >= len(images20[-1])):
                            #     imagemid2 = widL = int(len(images20[-1])/2)
                            #     imagemid2_2 = widL_2 = int(imagemid2*0.8)
                            #     imagemid3_3 = widL_3 = int(imagemid2*0.6)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

                        print("L1_lenim2:",len(images20[-1]),"imageid2",imagemid2,"lentest2:",len(test2),"width", len(LLw[-1]), widL,"liner22:",len(liner22),"slop2:",len(slop_d2),"imc2:",len(imc2), imc2[-1],"nem:",nem[-1])
                        
                        # images20[-1][imagemid2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TL1" + "_" + str(LLw[-1][widL]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images20[-1][imagemid2_2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TL2" + "_" + str(LLw[-1][widL_2]) + "-" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        # images20[-1][imagemid3_3].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TL3" + "_" + str(LLw[-1][widL_3]) + "-"+ time.strftime("%d-%m-%Y_%H-%M-%S") + ".jpg")
                        
                        images20[-1][imagemid2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-1][widL]) + "_" + str(nem[-1]) + "L1.jpg")
                        images20[-1][imagemid2_2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-1][widL_2]) + "_" + str(nem[-1]) + "L2.jpg")
                        images20[-1][imagemid3_3].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-1][widL_3]) + "_" + str(nem[-1]) + "L3.jpg")


                        # images20[-1][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "L1" + "_" + str(LLw[-1][widL]) + "_" + str(len(images20[-1])) + "-" + str(imagemid2) + ".jpg")
                        # images20[-1][imagemid2_2].save("outputs/" + str(nem[-1]) + "_" + "L2" + "_" + str(LLw[-1][widL_2]) + "_" + str(len(images20[-1])) + "-" + str(imagemid2_2) + ".jpg")
                        # images20[-1][imagemid3_3].save("outputs/" + str(nem[-1]) + "_" + "L3" + "_" + str(LLw[-1][widL_3]) + "_"+ str(len(images20[-1])) + "-" + str(imagemid3_3) + ".jpg")
                        nem[0]+=1

                        leennn = int(len(test2)-1)

                        if(len(test2)>1):
                            dipp1 = test2[-1]
                            dipp2 = test2[-2]
                            dipp3 = dipp1[0]
                            dipp4 = dipp2[0]
                            print(dipp3[0],dipp4[0])
                            if (dipp3[0]<dipp4[0]):
                            #if (dipp3[0]<dipp4[0]) | (dipp3[0]>dipp4[0]):
                                images20.clear()
                                test2.clear()
                                slop_d2.clear()
                                liner22.clear()
                                lol2.clear()
                                imc2.clear()
                                LLw.clear()
                                
                                
        
                                #print('full',len(images20))
    
                        else:
                            images20 = images20[:leennn]
                            test2 = test2[:leennn]
                            liner22 = liner22[:leennn]
                            slop_d2 = slop_d2[:leennn]
                            lol2 = lol2[:leennn]
                            imc2 = imc2[:leennn]
                            LLw = LLw[:leennn]
                            

                            #print('single',len(images20))
                        swap2 = 0

                if len(test2)>1:
                    itee3 = test2[-2]
                    itee4 = test2[-1]
                    itee5 = itee3[0]
                    itee6 = itee4[0]
                    sertt1 = itee3[1]
                    sertt2 = itee4[1]
                    
                    sertt11 = (itee5[0]+sertt1[0])/2
                    sertt22 = (itee6[0]+sertt2[0])/2
                    
                    # print("liner2[-1]:",(liner22[-1]),"2:",len(test2), int(liner22[-1])/len(test2))
                    #print(test2)
                    #print("sert11",sertt11,"ser11",sertt11,"itee6[1]",itee6[1],"itee5[1]",itee5[1],"itee6[0]",itee6[0],"itee5[0]",itee5[0])
                    if(sertt22>sertt11) | (abs(itee6[1]-itee5[1])>250) | (itee6[0]>itee5[0]):
                        if ocean == "left":
                            imagemid2 = widL = int(imc2[-2]*0.15)
                            imagemid2_2 = widL_2 = int(imagemid2*1.2)
                            imagemid3_3 = widL_3 = int(imagemid2*1.5)
                            
                    
                            if((imc2[-2] < 6) | (swap2 == 1)):
                                imagemid2 = widL = int(len(images20[-2])/1.5)
                                imagemid2_2 = widL_2 = int(imagemid2*1.1)
                                imagemid3_3 = widL_3  = int(imagemid2*0.9)
                                
                            
                            # if(imagemid2 >= len(images20[-2])):
                            #     imagemid2 = widL = int(len(images20[-2])/2)
                            #     imagemid2_2 = widL_2 = int(imagemid2*1.2)
                            #     imagemid3_3 = widL_3 = int(imagemid2*1.5)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                                

                        if ocean == "right":
                            imagemid2 = widL = int(imc2[-2]*0.85)
                            imagemid2_2 = widL_2 = int(imagemid2*0.8)
                            imagemid3_3 = widL_3 = int(imagemid2*0.6)
                    
                            if((imc2[-2] < 6) | (swap2 == 1)):
                                imagemid2 = widL = int(len(images20[-2])/1.5)
                                imagemid2_2 = widL_2 = int(imagemid2*0.9)
                                imagemid3_3 = widL_3 = int(imagemid2*1.1)
                            
                            # if(imagemid2 >= len(images20[-2])):
                            #     imagemid2 = widL = int(len(images20[-2])/2)
                            #     imagemid2_2 = widL_2 = int(imagemid2*0.8)
                            #     imagemid3_3 = widL_3 = int(imagemid2*0.6)
                            #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

                        

                        print("L2_lenim22:",len(images20[-2]),"imageid2",imagemid2,"lentest2:",len(test2),"width", len(LLw[-2]), widL,"liner22:",len(liner22),"slop2:",len(slop_d2),"imc2:",len(imc2), imc2[-2],"nem:",nem[-1])
            
                        # images20[-2][imagemid2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TLL1" + "_" + str(LLw[-2][widL]) + time.strftime("%d-%m-%Y_%H-%M-%S") +".jpg")
                        # images20[-2][imagemid2_2].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TLL2" + "_" + str(LLw[-2][widL_2]) + time.strftime("%d-%m-%Y_%H-%M-%S") +".jpg")
                        # images20[-2][imagemid3_3].save("/run/user/1000/gvfs/smb-share:server=mari_storage_1.local,share=recordings/Flir/boat_imgs/" + str(nem[-1]) + "_" + "TLL3" + "_" + str(LLw[-2][widL_3]) + time.strftime("%d-%m-%Y_%H-%M-%S") +".jpg")


                        images20[-2][imagemid2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-2][widL]) + "_" + str(nem[-1]) + "LL1.jpg")
                        images20[-2][imagemid2_2].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-2][widL_2]) + "_" + str(nem[-1]) + "LL2.jpg")
                        images20[-2][imagemid3_3].save("outputs/" + time.strftime("%m-%d-%Y_%H-%M-%S") + "_" + str(LLw[-2][widL_3]) + "_" + str(nem[-1]) + "LL3.jpg")

                        # images20[-2][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "LL1" + "_" + str(LLw[-2][widL]) + "_" + str(len(images20[-2])) + "-" + str(imagemid2) +".jpg")
                        # images20[-2][imagemid2_2].save("outputs/" + str(nem[-1]) + "_" + "LL2" + "_" + str(LLw[-2][widL_2]) + "_" + str(len(images20[-2])) + "-" + str(imagemid2_2) +".jpg")
                        # images20[-2][imagemid3_3].save("outputs/" + str(nem[-1]) + "_" + "LL3" + "_" + str(LLw[-2][widL_3]) + "_" + str(len(images20[-2])) + "-" + str(imagemid3_3) +".jpg")

                        nem[0]+=1

                        # images20.pop(-2)
                        # test2.pop(-2)
                        # liner22.pop(-2)
                        # slop_d2.pop(-2)
                        # lol2.pop(-2)
                        # imc2.pop(-2)
                        # LLw.pop(-2)

                        leennn = int(len(test2)-1)
                        #print("bb",len(images20[-2]),len(LLw[-2]))
                        images20 = images20[leennn:]
                        test2 = test2[leennn:]
                        liner22 = liner22[leennn:]
                        slop_d2 = slop_d2[leennn:]
                        lol2 = lol2[leennn:]
                        imc2 = imc2[leennn:]
                        LLw = LLw[leennn:]
                        
                        #print("AA",len(images20[-2]),len(LLw[-2]))
                        #print("dsingle",len(images2))
                        swap2 = 1
        
            
            cv2.line(frame, (left_border,0), (left_border,2160), (0, 0, 255), thickness=5, lineType=1)
            cv2.line(frame, (right_border,0), (right_border,2160), (0, 0, 255), thickness=5, lineType=1)

            cv.imshow(winName, frame)

end = time.strftime("%d-%m-%Y_%H-%M-%S")
fps.stop()

print(jstart)
print(end)

