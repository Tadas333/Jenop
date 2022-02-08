import cv2 as cv
from matplotlib.style import available
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import time
from test7 import CentroidTracker
import math
from statistics import mean
from PIL import Image



#Write down conf, nms thresholds,inp width/height
confThreshold = 0.5
#confThreshold = 0.4
nmsThreshold = 0.40
#nmsThreshold = 0.50
inpWidth = 416
inpHeight = 416
# inpWidth = 288
# inpHeight = 288

cap = cv2.VideoCapture('not_working.mp4')

left_border = 400
right_border = 2600

images11 = 0
images22 = 0

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

test1 = []
test2 = []
wipe = []

nem = [0]

#Load names of classes and turn that into a list
classesFile = "obj.names"
classes = None


#Model configuration
modelConf = 'yolov4-tiny_custom.cfg'
modelWeights = 'yolov4-tiny_custom_final.weights'

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




fps = FPS().start()

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

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

                if dist < 200:
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

        images1.clear()

        

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

        images2.clear()


    cords.clear()
    cords1.clear()
    cords2.clear()
    countX.clear()
    countY.clear()
    liner1.clear()
    liner2.clear()

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

            imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            im = Image.fromarray(imageRGB)
            im1 = im.crop((left-80, top-80, left+width+80, top+height+80))
            imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            label = '%.2f' % confidences[i]
            confy = '%.2f' % (confidences[i] * 100)
       
         
            if len(countX)>2:
                if(abs(countX[-1]-countX[-2])<200) & (abs(countY[-1]-countY[-2])<100):
                    #if(((abs(countX[-1]-right_border)>300)) & (abs(countX[-1]-left_border)>300)):
                    if((countX[-1]<right_border) & (countX[-1]>left_border)):
                    
                        cords.append((centerX,centerY))

                        if( countX[-1]>countX[-2]>countX[-3]):
                            cords1.append((centerX,centerY))
                            images1.append(im1)
                            liner1.append('1')
          
                        if( countX[-1]<countX[-2]<countX[-3]):
                            cords2.append((centerX,centerY))
                            images2.append(im1)
                            liner2.append('1')

                        if((countX[-1]>countX[-2]>countX[-3]) & (abs(centerX-right_border)<50)) | ((countX[-1]<countX[-2]<countX[-3]) & (abs(centerX-left_border)<50)):
                            do()

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
                cv.putText(frame, str(confy), (left , top), font, 4, (255,255,255), 2) # count yazısı
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    
    
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
 
    ret = cap.grab() #grab frame
    i=i+1 #increment counter
    iei = [i]
   
    if i % 2 == 0: # display only one third of the frames, you can change this parameter according to your needs
        ret, frame = cap.retrieve()

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
            
            cv2.line(frame, c,n, (0, 255, 255), thickness=3, lineType=1)


        for item in test1:
            cv2.line(frame, item[0],item[1], (220, 50, 255), thickness=4, lineType=1)
            for iii, element in enumerate(test1): 
                pp = test1[iii-1] if iii > 0 else 0
                cc = element 
                nn = test1[iii+1] if iii < len(test1)-1 else 0 

                if (cc == 0) | (nn == 0):
                    break
                
                ccc = cc[1]
                nnn = nn[0]

                if (ccc[0]<nnn[0]):
                    cv2.line(frame, ccc,nnn, (220, 50, 255), thickness=4, lineType=1)
                    
            
            if(len(test1)>1):
                if test1[-1] != last_v1:
                    mam1 = test1[-1]
                    mam2 = test1[-2]
                    mamm3 = mam1[0]
                    mamm4 = mam2[1]
                    if mamm3[0]>mamm4[0]:   
                        last_v1 = test1[-1]
                        samp1 = sum(imc1)
                        samp11 = images10[-2] + images10[-1]
                        imc1.pop()
                        images10.pop()
                        imc1.append(samp1)
                        images10.append(samp11)
                        


            if(len(liner11)>0) & (len(slop_d1)>0) & (len(lol1)>0):
                finder1 = int(liner11[-1])/10
                ite2 = item[1] 
                catcher = (slop_d1[-1]/liner11[-1]) + ((slop_d1[-1]/liner11[-1])/0.9)
                
                if ((iei[-1]-lol1[-1])>catcher) | (abs(ite2[0]-right_border)<100): 
                    if len(images10[-1]) == imc1[-1]:
                        imagemid = int(imc1[-1]*0.1)
                    else:
                        imagemid = int(len(images10[-1])-imc1[-1]+1)
                        
                    if((imagemid == 0) | (swap1 == 1)):
                        imagemid = int(len(images10[-1])/1.5)

                    #print("lenim1:",len(images10[-1]),"imageid1",imagemid,"lentest1:",len(test1),"liner11:",len(liner11),"slop1:",len(slop_d1),"imc1:",len(imc1), imc1[-1],"nem:",nem[-1])
         
                    images10[-1][imagemid].save("outputs/" + str(nem[-1]) + "_" + "right3" + "_" + ".jpg")
                    nem[0]+=1
                
                    lennn = int(len(test1)-1)

                    if(len(test1)>1):
                        dip1 = test1[-1]
                        dip2 = test1[-2]
                        dip3 = dip1[0]
                        dip4 = dip2[0]
                        #print(dip3[0],dip4[0])
                        if dip3[0]>dip4[0]:
                            
                            images10.clear()
                            test1.clear()
                            slop_d1.clear()
                            liner11.clear()
                            lol1.clear()
                            imc1.clear()
                            
                            #print('full',len(images1))
  
                    else:
                        images10 = images10[:lennn]
                        test1 = test1[:lennn]
                        liner11 = liner11[:lennn]
                        slop_d1 = slop_d1[:lennn]
                        lol1 = lol1[:lennn]
                        imc1 = imc1[:lennn]
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
                finder1 = int(liner11[-1])/len(test1)

                if((sert11>sert22) | (abs(ite6[1]-ite5[1])>150)):
                    imagemid = int(len(images10[-1])/finder1)
                    #print('im11',len(images1),'test1',len(test1),"imageid",imagemid,"finder", finder1,"use1", use1,"nem",nem[-1])
                    images10[-1][imagemid].save("outputs/" + str(nem[-1]) + "_" + "right2" + "_" + ".jpg")
                    nem[0]+=1

                    lennn = int(len(test1)-1)               
                
                    images10 = images10[:lennn]
                    test1 = test1[:lennn]
                    liner11 = liner11[:lennn]
                    slop_d1 = slop_d1[:lennn]
                    lol1 = lol1[leennn:]
                    imc1 = imc1[:lennn]
                    swap1 = 1

                        


        for itemm in test2:
            cv2.line(frame, itemm[0],itemm[1], (255, 128, 128), thickness=4, lineType=1) 
            for iiii, element in enumerate(test2): 
                ppp = test2[iiii-1] if iiii > 0 else 0
                ccc = element 
                nnn = test2[iiii+1] if iiii < len(test2)-1 else 0 

                if (ccc == 0) | (nnn == 0):
                    break
                
                cccc = ccc[1]
                nnnn = nnn[0]

                if (cccc[0]>nnnn[0]):
                    cv2.line(frame, cccc,nnnn, (255, 128, 128), thickness=4, lineType=1)

            if(len(test2)>1):
                if test2[-1] != last_v2:
                    maam1 = test2[-1]
                    maam2 = test2[-2]
                    maamm3 = maam1[0]
                    maamm4 = maam2[1]
                    if maamm3[0]>maamm4[0]:   
                        last_v2 = test2[-1]
                        samp2 = sum(imc2)
                        samp22 = images20[-2] + images20[-1]
                        imc2.pop()
                        images20.pop()
                        imc2.append(samp2)
                        images20.append(samp22)
                        

            if(len(liner22)>0) & (len(slop_d2)>0) & (len(lol2)>0):
                finder2 = int(liner22[-1])/10
                itee2 = itemm[1]
                catcher2 = (slop_d2[-1]/liner22[-1]) + ((slop_d2[-1]/liner22[-1])/0.9)
                #catcher2 = ((slop_d2[-1]/liner22[-1]) + 50)

                if ((iei[-1]-lol2[-1])>catcher2) | (abs(itee2[0]-left_border)<100):
                    if len(images20[-1]) == imc2[-1]:
                        imagemid2 = int(imc2[-1]-2)
                    else:
                        imagemid2 = int(len(images20[-1])-imc2[-1]+1)
            
                    if((imagemid2 == 0) | (swap2 == 1)):
                        imagemid2 = int(len(images20[-1])/1.5)

                    #print("lenim2:",len(images20[-1]),"imageid2",imagemid2,"lentest2:",len(test2),"liner22:",len(liner22),"slop2:",len(slop_d2),"imc2:",len(imc2), imc2[-1],"nem:",nem[-1])
                    
                    images20[-1][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "left3" + "_" + ".jpg")
                    nem[0]+=1

                    leennn = int(len(test2)-1)

                    if(len(test2)>1):
                        dipp1 = test2[-1]
                        dipp2 = test2[-2]
                        dipp3 = dipp1[0]
                        dipp4 = dipp2[0]
                        #print(dipp3[0],dipp4[0])
                        if dipp3[0]<dipp4[0]:
                            images20.clear()
                            test2.clear()
                            slop_d2.clear()
                            liner22.clear()
                            lol2.clear()
                            imc2.clear()
                            images22 = 0
    
                            #print('full',len(images20))
 
                    else:
                        images20 = images20[:leennn]
                        test2 = test2[:leennn]
                        liner22 = liner22[:leennn]
                        slop_d2 = slop_d2[:leennn]
                        lol2 = lol2[:leennn]
                        imc2 = imc2[:leennn]

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
                finder2 = int(liner22[-1])/len(test2)


                if((sertt22>sertt11) | (abs(itee6[1]-itee5[1])>150)):
                    if len(images20[-2]) == imc2[-2]:
                        imagemid2 = int(imc2[-2]/2)
                    else:
                        imagemid2 = int(len(images20[-2])-imc2[-2]+1)
                        
                    if((imagemid2 == 0) | (swap2 == 1)):
                        imagemid2 = int(len(images20[-2])/1.5)
                  
                    
                    #print("lenim2:",len(images20[-2]),"imageid2",imagemid2,"lentest2:",len(test2),"liner22:",len(liner22),"slop2:",len(slop_d2),"imc2:",len(imc2), imc2[-2],"nem:",nem[-1])
        
                    images20[-2][imagemid2].save("outputs/" + str(nem[-1]) + "_" + "left2" + "_" + ".jpg")
                    nem[0]+=1

                    leennn = int(len(test2)-1)

                    images20 = images20[leennn:]
                    test2 = test2[leennn:]
                    liner22 = liner22[leennn:]
                    slop_d2 = slop_d2[leennn:]
                    lol2 = lol2[leennn:]
                    imc2 = imc2[leennn:]
                    #print("dsingle",len(images2))
                    swap2 = 1
    
         
        cv2.line(frame, (left_border,0), (left_border,2160), (0, 0, 255), thickness=3, lineType=1)
        cv2.line(frame, (right_border,0), (right_border,2160), (0, 0, 255), thickness=3, lineType=1)

        cv.imshow(winName, frame)
fps.stop()

