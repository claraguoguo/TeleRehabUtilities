import glob
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os

# The difference between the y-coordinates of left-hand and right-hand
#  should not exceed HANDS_DIFF_THRESHOLD
HANDS_DIFF_THRESHOLD = 50
NUM_REPETITION = 5
FILE_PATH = '/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/KiMoRe_skeletal_txt_files_all_joints'
FEATURES_FILE_PATH = '/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/KiMoRe_skeletal_features'
FEATURES_PLOTS_PATH = '/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/KiMoRe_skeletal_features_plots'


def plot_body_joints(data, peaks_index, features, video_name):
    fig = plt.figure()

    for index, counter in zip(peaks_index, range(NUM_REPETITION)):
        ax = fig.add_subplot(2, 3, counter + 1)
        feature = features[counter]
        frame = data[index, :]
        xs = np.array([])
        ys = np.array([])
        for i in range(0, len(frame)):
            if i % 2:
                ys = np.append(ys, frame[i])
            else:
                xs = np.append(xs, frame[i])

        non_zero_indices = np.nonzero(xs)[0]
        non_zero_xs = xs[non_zero_indices]
        non_zero_ys = ys[non_zero_indices]

        ax.plot(non_zero_xs, non_zero_ys, 'bo')
        # ax.plot(xs, ys, 'bo')

        for ind in non_zero_indices:
        # for ind in range(25):
            label = "{}".format(ind)

            ax.annotate(label,  # this is the text
                         (xs[ind], ys[ind]),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        ax.invert_yaxis()
        ax.set_title('timestamp: {0} \nleft_elbow_angle: {1:.1f}   right_elbow_angle: {2:.1f} '
                     '\nhand_distance: {3:.1f}   torso_area: {4:.1f}'.\
                     format(index, feature[0], feature[1], feature[2], feature[3]))
        # ax.show(block=False),
        # ax.pause(1)
        # ax.close()

    plt.tight_layout()
    plt.suptitle(video_name)
    # plt.show(block=False)
    # plt.pause(1)
    plt.savefig(os.path.join(FEATURES_PLOTS_PATH, video_name + '.png'))
    plt.close()


def get_peaks_index(data, num_peaks):

    '''
    From testing, if there is less than 64 frames, then split data into 4 sections won't work well.
    The local peaks found were not accurate.

    '''

    # split into data into num_peaks number of sections and find 1 peak in each section
    split_data = np.array_split(data, num_peaks)

    index_count = 0
    peaks_index = []
    for array in split_data:
        ind = np.argmin(array)
        peaks_index.append(ind + index_count)
        index_count += len(array)

    return peaks_index

# Source: https://stackoverflow.com/a/13849249/8257793
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            90
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            180
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    radian = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    degree = np.degrees(radian)
    return degree

# Implementation of Shoelace formula.
# Source: https://stackoverflow.com/a/30408825/8257793
def compute_poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def compute_features(peaks_index, data):
    '''

    :param peaks_index: 4 timestamps when the hands are raised to the highest positions
    :param data: 25 body joints extracted from "GOOD" frames, which have the 10 upper body joints
    :return: [left_elbow_angle, right_elbow_angle, hand_distance, torso_area]
    '''
    features = np.empty(shape=(0,4))
    for index in peaks_index:
        frame = data[index, :]
        ''' 
        Left arm:
        pt_2: left shoulder
        pt_3: left elbow
        pt_4: left hand
        
        pt_9: left hip
        '''
        pt_2 = np.array([frame[2 * 2], frame[2 * 2 + 1]])
        pt_3 = np.array([frame[3 * 2], frame[3 * 2 + 1]])
        pt_4 = np.array([frame[4 * 2], frame[4 * 2 + 1]])
        pt_9 = np.array([frame[9 * 2], frame[9 * 2 + 1]])

        ''' 
        Right arm:
        pt_5: right shoulder
        pt_6: right elbow
        pt_7: right hand
        
        pt_12: right hip
        '''
        pt_5 = np.array([frame[5 * 2], frame[5 * 2 + 1]])
        pt_6 = np.array([frame[6 * 2], frame[6 * 2 + 1]])
        pt_7 = np.array([frame[7 * 2], frame[7 * 2 + 1]])
        pt_12 = np.array([frame[12 * 2], frame[12 * 2 + 1]])

        # vec_32: vector from pt_3 to pt_2
        vec_32 = pt_2 - pt_3
        vec_34 = pt_4 - pt_3
        left_elbow_angle = angle_between(vec_32, vec_34)

        vec_65 = pt_5 - pt_6
        vec_67 = pt_7 - pt_6
        right_elbow_angle = angle_between(vec_65, vec_67)

        # Compute the distance between left hand and right hand
        vec_47 = pt_7 - pt_4
        hand_distance = np.linalg.norm(vec_47)

        # Compute the torso area which the area enclosed by body points: 2, 5, 9, 12
        torso = np.array([pt_2, pt_5, pt_12, pt_9])
        torso_area = compute_poly_area(torso[:, 0], torso[:, 1])

        curr_features = np.asarray([left_elbow_angle, right_elbow_angle, hand_distance, torso_area])

        features = np.vstack((features, curr_features))
    return features

def main():
    for filepath in glob.glob(FILE_PATH + '/*.txt', recursive=True):
        # filepath = FILE_PATH + "/GPP_Stroke_S_ID6_Es1_rgb_Blur_rgb040716_112230.txt"

        # video has shape [frames x num_joints]
        data = np.loadtxt(filepath, delimiter=',')

        # show the point plot
        # show_plots(data[0])

        # left hand : point 4
        left_hand_x = data[:, 4 * 2]
        left_hand_y = data[:, 4 * 2 + 1]

        # right hand : point 7
        right_hand_x = data[:, 7 * 2]
        right_hand_y = data[:, 7 * 2 + 1]

        # Subjects were asked to repeat each exercise consecutively 5 times
        # Find 5 local min for y-coordinates (local min == hand rise above head)

        sum_left_right = left_hand_y + right_hand_y
        diff_left_right = abs(left_hand_y - right_hand_y)

        peaks_index = get_peaks_index(sum_left_right, NUM_REPETITION)  # Find 5 peaks
        print(peaks_index)

        # Features = [num_peaks x num_features]
        # 1 feature = [left_elbow_angle, right_elbow_angle, hand_distance, torso_area]
        features = compute_features(peaks_index, data)

        video_name = os.path.basename(filepath).split('.')[0]
        # plot_body_joints(data, peaks_index, features, video_name)

        # Save the features
        np.savetxt(os.path.join(FEATURES_FILE_PATH, video_name + '.txt'), features, delimiter=',', fmt='%1.3f')

if __name__ == '__main__':
    main()
