import glob
from scipy.signal import find_peaks
import numpy as np
import os
import json

EXERCISE_TYPE = 'Es5'
FILE_ROOT = f"/Users/Clara_1/Google Drive/KiMoRe_skeletal_{EXERCISE_TYPE}/"
OUTPUT_FOLDER = f"/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/{EXERCISE_TYPE}/KiMoRe_skeletal_txt_files_all_joints_2/"
UPPER_BODY_INDEX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 12])
LOWER_BODY_INDEX = np.array([10, 11, 13, 14, 19, 20, 21, 22, 23, 24])
LEG_INDEX = np.array([10, 13])

VALID_JOINTS_INDEX = np.append(UPPER_BODY_INDEX, LEG_INDEX)
# VALID_JOINTS_INDEX = UPPER_BODY_INDEX

TOTAL_BODY_JOINTS = 25

def get_valid_points(joints):
    '''
    :param joints: body joints
    :return: A list of valid body joints in (x, y)
    '''
    output = []
    for index in VALID_JOINTS_INDEX:
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
    missing_valid_points = [i for i in VALID_JOINTS_INDEX if joints[i*3] == 0]

    # missing_upper_body = [i for i in UPPER_BODY_INDEX if joints[i*3] == 0]
    # missing_lower = [i for i in LOWER_BODY_INDEX if joints[i*3] == 0]
    # if len(missing_upper_body) == 0 and len(missing_lower) > 3:
    #     print("missing_lower:{} {}".format(len(missing_lower), missing_lower))
    #     # print(len(missing_lower))

    # Return True if none of the key points in VALID_JOINTS_INDEX is missing
    return len(missing_valid_points) == 0


def process_JSON():
    try:
        os.mkdir(OUTPUT_FOLDER)
    except OSError:
        print("Creation of the directory %s failed!" % OUTPUT_FOLDER)
        # raise
    for rootdir, dirs, files in os.walk(FILE_ROOT):
        for subdir in dirs:
            video_path = os.path.join(rootdir, subdir)

            all_valid_joints = np.empty((0, len(VALID_JOINTS_INDEX) * 2))
            all_joints = np.empty((0, TOTAL_BODY_JOINTS * 2))

            total_frames = 0
            valid_frames = 0
            for filepath in glob.glob(video_path + '/*.json', recursive=True):
                total_frames += 1
                data = json.load(open(filepath))
                # body_joints is a list of size 25
                if (len(data['people']) != 0):
                    body_joints = data['people'][0]['pose_keypoints_2d']
                    if (is_good_frame(body_joints)):
                        # all_valid_joints only has selected joints (i.e. 10 upper body joints), this data is clean!
                        all_valid_joints = np.vstack((all_valid_joints, np.asarray(get_valid_points(body_joints))))

                        # all_joints has all 25 joints, this includes zero values (missing points)
                        all_joints = np.vstack((all_joints, np.asarray(get_all_points(body_joints))))
                        valid_frames += 1

            # Save joints extracted from valid frames into txt file
            output_txt_file = os.path.join(OUTPUT_FOLDER, subdir + ".txt")
            np.savetxt(output_txt_file, all_joints, delimiter=",", fmt='%1.3f')
            print("video: {}  Total frames:{}  Valid frames:{}  Bad frames: {} ".format(
                subdir, total_frames, valid_frames, total_frames-valid_frames))



    # for filepath in glob.glob(FILE_ROOT):
    #     print(filepath)




if __name__ == '__main__':
    '''
    Extract joints from the JSON files and put them into text file
    '''
    process_JSON()
