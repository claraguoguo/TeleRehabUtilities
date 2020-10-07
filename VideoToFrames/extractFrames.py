import cv2     # for capturing videos
import math   # for mathematical operations

import os
# import matplotlib.pyplot as plt    # for plotting the images
# import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
# import numpy as np    # for mathematical operations
# from keras.utils import np_utils
# from skimage.transform import resize   # for resizing image
#


# extracting 1 frame per second
def extractFramesFromVideo(videoPath, outputImagePath):
    count = 0
    cap = cv2.VideoCapture(videoPath)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    videoName = os.path.splitext(os.path.basename(videoPath))[0]
    outputPath = outputPath = os.path.join(outputImagePath, videoName)
    if (os.path.exists(outputPath) != True):
        os.mkdir(outputPath)
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frameName = "frame%d.jpg" % count;
            count += 1
            cv2.imwrite(os.path.join(outputPath, frameName), frame)
    cap.release()
    print("Done!")



if __name__ == '__main__':
    # input: video data directory
    directoryPath = './dataset/v1.mp4'

    # output: extracted frame directory
    outputImagePath =  './output/'

    if (os.path.exists(outputImagePath) != True):
        os.mkdir(outputImagePath)

    extractFramesFromVideo(directoryPath, outputImagePath)