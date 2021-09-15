import cv2
import numpy as np 
import glob, os 
from datetime import datetime
import time, sys

#path1 = 'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution'
#path1 = 'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/vid1.mp4'

path1 = r'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/outputs'
path2 = r'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/changed/'
path3 = r'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/changed'


while(1):

    for vinput in os.listdir(path1): 
        if vinput.endswith('.mp4'):

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
            

            #print(dt_string)
            
            new_name =  dt_string + ".mp4"
            os.rename(os.path.join(path1, vinput), os.path.join(path2, path2 + new_name))
            now1 = time.time()
            for f in os.listdir(path3):
                f = os.path.join(path3, f)
                if os.stat(f).st_mtime < now1 - 50:
                    os.remove(os.path.join(path3, f))
                    time.sleep(10)
    


