from PIL import Image
import os, os.path
import numpy as np
import cv2

path = "/home/lab/Mass_Dataset_All/"
path_keep = "/home/lab/Pushing_Preprocessed/"
valid_images = [".png"]

count = 0

def preprocess(img):
    #print(img)
#    img_copy = np.zeros((84, 84, 3))
#    n_blue = 0
#    n_green = 0
#    for i in range(0, len(img)):
#        for j in range(0, len(img[0])):
#            if img[i][j][2] < 30 and img[i][j][1] < 30 and img[i][j][0] > 90:
#                n_blue += 1
#            if img[i][j][2] < 30 and img[i][j][1] > 90 and img[i][j][0] < 30:
#                n_green += 1
#    print(n_blue, n_green)

    n_blue = np.sum(np.all(((img <= [10000, 30, 30]) & (img >= [90, 0, 0])),axis=-1))
    n_green = np.sum(np.all(((img <= [30, 1000000, 30]) & (img >= [0, 90, 0])),axis=-1))

    #n_green = np.sum(np.all(img <= [30, 90, 30],axis=-1))
#    print(n_blue, n_green)
    if n_blue > 100 and n_green > 100:
        return 1
    #cv2.imwrite('a.png', img_copy)
    #cv2.imwrite('b.png', img)
    return 0

for dirpath, dirnames, filenames in os.walk(path):
    i = 0
    for filename in [f for f in filenames if f.endswith(".png")]:
        #print(os.path.join(dirpath, filename))
        img_1 = cv2.imread(dirpath + "/" + str(i) + "_0.png")
        img_2 = cv2.imread(dirpath + "/" + str(i) + "_1.png")
        if img_1 is not None and img_2 is not None:
            np_arr = np.load(dirpath + "/" + str(i) + ".npy", allow_pickle=True)
            keep = preprocess(img_1) and preprocess(img_2)
            if keep:
                cv2.imwrite(path_keep + str(count) + "_0.png", img_1)
                cv2.imwrite(path_keep + str(count) + "_1.png", img_2)
                np.save(path_keep + str(count) + ".npy", np_arr)
                #print(np_arr)
                count += 1
        i += 1
