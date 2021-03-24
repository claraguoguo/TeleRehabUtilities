import cv2     # for capturing videos
import math   # for mathematical operations

import os
import matplotlib.pyplot as plt    # for plotting the images
import numpy as np
# import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
# import numpy as np    # for mathematical operations
# from keras.utils import np_utils
# from skimage.transform import resize   # for resizing image
#

import subprocess



def getVideoLength(videoRootDir, exerciseType):
    count = 0
    videoLengths = []
    videoNames = []
    for root, dirs, files in os.walk(videoRootDir):

        if (root.find(exerciseType) != -1):
            for file in files:
                if file.endswith(".mp4"):
                    filename = os.path.join(root, file)

                    # get the length of each video
                    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                             "format=duration", "-of",
                                             "default=noprint_wrappers=1:nokey=1", filename],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                    videoLengths.append(float(result.stdout))
                    videoNames.append(filename)
                    count += 1


    # Find the videos with MAX length and MIN length
    minLength = min(videoLengths)
    maxLength = max(videoLengths)
    averageLength = -1


    if (len(videoLengths) > 0):
        averageLength = sum(videoLengths) / len(videoLengths)
        print("Exercise: {}\nTotal Number of Videos: {}   Average Length: {:.2f}   MAX: {:.2f}   MIN: {:.2f} "
            .format(exerciseType, count, averageLength, maxLength, minLength))
    else:
        print("No video found!")


    # # Draw histogram of the data
    # plt.hist(videoLengths, bins=10, color='c', edgecolor='k', alpha=0.65)
    # plt.xlabel("Video Length")
    # plt.ylabel("Frequency")
    # plt.xticks(np.arange(math.floor(minLength), math.ceil(maxLength), 5))
    #
    # # write the mean value on the plot
    # plt.axvline(averageLength, color='k', linestyle='dashed', linewidth=1)
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(averageLength*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(averageLength))

    # plt.show()
    # plt.imsave()

    print(videoNames[videoLengths.index(minLength)])
    print(videoNames[videoLengths.index(maxLength)])
    return Tru


# extracting 1 frame per second




if __name__ == '__main__':
    # input: video data directory
    directoryPath = './dataset/v1.mp4'def extractFramesFromVideo(videoPath, outputImagePath):
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
            frameName = "frame%d.jpg" % count
            count += 1
            cv2.imwrite(os.path.join(outputPath, frameName), frame)
    cap.release()
    print("Done!")

    # output: extracted frame directory
    outputImagePath =  './output2/'

    if (os.path.exists(outputImagePath) != True):
        os.mkdir(outputImagePath)

    # extractFramesFromVideo(directoryPath, outputImagePath)


    exerciseType = "Es4"
    videoPath = "/Users/Clara/Desktop/KiMoRe/RGB/"
    getVideoLength(videoPath, "Es4")

