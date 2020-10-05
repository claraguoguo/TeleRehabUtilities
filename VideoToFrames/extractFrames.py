import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
# import numpy as np    # for mathematical operations
# from keras.utils import np_utils
# from skimage.transform import resize   # for resizing image
#

def extractFramesFromVideo(videoPath, fps):
    count = 0
    cap = cv2.VideoCapture(videoPath)  # capturing the video from the given path
    frameRate = cap.get(fps)  # frame rate
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = "frame%d.jpg" % count;
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Done!")


if __name__ == '__main__':
    directoryPath = '/Users/Clara/Desktop/KiMoRe-Full'
    extractFramesFromVideo(directoryPath)