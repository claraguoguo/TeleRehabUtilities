import glob
from scipy.signal import find_peaks
import numpy as np
import os
import json

FILE_ROOT = "/Users/Clara_1/Google Drive/KiMoRe_skeletal/"
OUTPUT_FILE = "/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/KiMoRe_skeletal_txt_files_all_joints/"
UPPER_BODY_INDEX = np.array([1,2,3,4,5,6,7,8,9,12])
TOTAL_BODY_JOINTS = 25

def get_valid_points(joints):
    '''
    :param joints: body joints
    :return: A list of valid body joints in (x, y)
    '''
    output = []
    for index in UPPER_BODY_INDEX:
        x = joints[index*3]
        y = joints[index*3 + 1]
        if (x == 0 or y == 9):
            print(1)
        output.extend((x, y))
    return output


def get_all_points(joints):
    '''
    :param joints: body joints
    :return: A list of all body joints in (x, y)
    '''
    output = []
    for index in range(TOTAL_BODY_JOINTS):
        x = joints[index*3]
        y = joints[index*3 + 1]
        output.extend((x, y))
    return output

def is_good_frame(joints):
    missing_valid_points = [i for i in UPPER_BODY_INDEX if joints[i*3] == 0]

    # Return True if none of the key points in UPPER_BODY_INDEX is missing
    return len(missing_valid_points) == 0


def process_JSON():
    for rootdir, dirs, files in os.walk(FILE_ROOT):
        for subdir in dirs:
            video_path = os.path.join(rootdir, subdir)

            all_valid_joints = np.empty((0, len(UPPER_BODY_INDEX) * 2))
            all_joints = np.empty((0, TOTAL_BODY_JOINTS * 2))

            total_frames = 0
            for filepath in glob.glob(video_path + '/*.json', recursive=True):
                data = json.load(open(filepath))
                # body_joints is a list of size 25
                if (len(data['people']) != 0):
                    body_joints = data['people'][0]['pose_keypoints_2d']
                    if (is_good_frame(body_joints)):
                        # all_valid_joints only has selected joints (i.e. 10 upper body joints), this data is clean!
                        all_valid_joints = np.vstack((all_valid_joints, np.asarray(get_valid_points(body_joints))))

                        # all_joints has all 25 joints, this includes zero values (missing points)
                        all_joints = np.vstack((all_joints, np.asarray(get_all_points(body_joints))))
                        total_frames += 1

            # Save joints extracted from valid frames into txt file
            output_txt_file = os.path.join(OUTPUT_FILE, subdir + ".txt")
            np.savetxt(output_txt_file, all_joints, delimiter=",", fmt='%1.3f')
            print("video: {}  Total frames:{} ".format(subdir, total_frames))



    # for filepath in glob.glob(FILE_ROOT):
    #     print(filepath)




if __name__ == '__main__':
    '''
    Extract joints from the JSON files and put them into text file
    '''
    process_JSON()
