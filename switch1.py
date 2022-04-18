import time
import cv2
import sys
import numpy as np
import subprocess 
#from subprocess import call 


start_count = 0

# proc1 = subprocess.Popen("C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/boat_only/outputs/", shell = True)
# time.sleep(2.0)
# print 'proc1 = ', proc1.pid
# subprocess.Popen.kill(proc1)

#subprocess.run([sys.executable, "-c", "raise ValueError('oops')"], check=True)

#proc1.kill()

# cap = cv2.VideoCapture('test_videos/switch.mp4')
# success,image = cap.read()
# while success:
#     print("in")
till = 0
still = 0
start_time = time.time() 

while True:
    
    start_count = time.time()
    #print(start_count/60,start_time/60)
    #print(count)
    if start_count > (start_time + 5):
        dim=20
        thresh=0.7
        # if till == 0:
        #     cap = cv2.VideoCapture('test_videos/dark.mp4')
        #     till = 1
        # else:
        #     cap = cv2.VideoCapture('test_videos/switch.mp4')
        cap = cv2.VideoCapture("rtspsrc location=rtsp://admin:Jenoptik33478@q192.168.1.220/h264/stream3 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
        success,image = cap.read()
        #while success:
        for x in range(1):
            #print("in")
            frame = cv2.resize(image, (dim, dim))
            # Convert color space to LAB format and extract L channel
            L, A, B = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
            # Normalize L channel by dividing all pixel values with maximum pixel value
            L = L/np.max(L)
            # cap.release()
            # cv2.destroyAllWindows()
            # Return True if mean is greater than thresh else False
            print("Mean", np.mean(L),"thresh", thresh, still)
            if np.mean(L) > thresh:
                #print("light")
                if still == 2:
                    print("set1")
                    print("mean", np.mean(L),"thresh", thresh, still)
                    proc2.kill()
                    still = 0

                if still == 1:
                    break

                if still == 0:
                    print("set11")
                    print("mean", np.mean(L),"thresh", thresh, still)
                    proc1 = subprocess.Popen(['python3', '/home/mari2/Desktop/switch_1/test16.py'])
                    proc11 = subprocess.Popen(['python3', '/home/mari2/Desktop/switch_1/all_det3.py'])
                    # proc1 = subprocess.Popen(['python3', 'C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/switch_1/test16.py'])
                    # proc11 = subprocess.Popen(['python3', 'C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/switch_1/all_det3.py'])

                    still = 1
            else:
                if still == 0:
                    #proc1.kill()
                    print("set2")
                    print("mean", np.mean(L),"thresh", thresh, still)
                    #proc2 = subprocess.Popen(['python3', 'C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/switch_1/test16T.py'])
                    proc2 = subprocess.Popen(['python3', '/home/mari2/Desktop/switch_1/test16T.py'])
                    still = 2
                if still == 1:
                    print("set22")
                    print("mean", np.mean(L),"thresh", thresh, still)
                    proc1.kill()
                    proc11.kill()
                    #proc2 = subprocess.Popen(['python3', 'C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/switch_1/test16T.py'])
                    proc2 = subprocess.Popen(['python3', '/home/mari2/Desktop/switch_1/test16T.py'])
                    still = 2


                #print("dark")  
        start_time = time.time()    
        cap.release()
        cv2.destroyAllWindows()
        start = time.time() 