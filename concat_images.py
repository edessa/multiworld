import os, os.path
import numpy as np
import cv2
from PIL import Image

path_keep = "/home/lab/Pushing_Preprocessed/"
path_out = "/home/lab/Input_Images/"
count = 0

def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = (224, 224)
    layer = Image.new('RGB', size, (0,0,0))
    layer.paste(img, tuple(map(lambda x:int((x[0]-x[1])/2), zip(size, img.size))))
    return layer

for count in range(136323):
    img_1 = cv2.imread(path_keep + str(count) + "_0.png")
    img_2 = cv2.imread(path_keep + str(count) + "_1.png")
    img = cv2.vconcat([img_1, img_2])
    img = Image.fromarray(img)
    square_img = white_bg_square(img)
    square_img.resize((224, 224), Image.ANTIALIAS)
    square_img.save(path_out + str(count) + ".png")
    #cv2.imwrite(path_out + str(count) + ".png", img)
