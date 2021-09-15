import cv2
import numpy as np 
import glob, os 
from datetime import datetime

#path1 = 'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution'
#path1 = 'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/vid1.mp4'

path1 = r'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution'
path2 = r'C:/Users/tadas.orentas/Desktop/PROJECTS/Vector_Streaming_solution/'



#for vinput in glob.glob(os.path.join(path1, '*.mp4')):
for vinput in os.listdir(path1): 
    if vinput.endswith('.mp4'):

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        

        #print(dt_string)
        
        new_name =  dt_string + ".mp4"
        os.rename(vinput, path2 + new_name)
    


