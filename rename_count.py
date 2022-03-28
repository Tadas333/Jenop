import imutils

import cv2
import numpy as np
from PIL import Image
import glob, os
from decimal import Decimal
import fileinput
import shutil

path = 'pic1.jpeg'

#os.chdir("C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Image_variator/test1")
os.chdir("C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/boat_only/outputs")
#c = "C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Image_variator/test1"
#p = "C:/Users/tadas.orentas/Desktop/PROJECTS/Maritime/Image_variator/test1"





count = 0


for file in glob.glob("*.JPG"):
    
    img = cv2.imread(file)
    
    cv2.imwrite(str(count) + ".jpg", img)

    os.remove(file)

    count += 1

    cv2.waitKey(0) & 0xFF == ord('q')

