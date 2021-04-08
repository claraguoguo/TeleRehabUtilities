import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# The difference between the y-coordinates of left-hand and right-hand
#  should not exceed HANDS_DIFF_THRESHOLD

CURR_FEATURES = ['left_elbow_angle', 'right_elbow_angle', 'hand_dist_ratio', 'torso_tilted_angle', 'hand_tilted_angle', 'elbow_angles_diff']
HANDS_DIFF_THRESHOLD = 50
NUM_REPETITION = 5

FILE_PATH = '/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/KiMoRe_skeletal_txt_files_all_joints'
OUTPUT_ROOT = '/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/'


def plot_body_joints(data, peaks_index, features, video_name, feature_plots_output):
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

        # Do not show (0,0) points
        ax.plot(non_zero_xs, non_zero_ys, 'bo')
        for ind in non_zero_indices:

        # Show (0,0) points
        # ax.plot([0]+xs, [0]+ys, 'bo')
        # for ind in range(25):
            label = "{}".format(ind)

            ax.annotate(label,  # this is the text
                         (xs[ind], ys[ind]),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        ax.invert_yaxis()

        title = f'timestamp: {index}'
        for i, feat in enumerate(CURR_FEATURES):
            title += ' {0}:{1:.1f}'.format(feat, feature[i])

        ax.set_title(title)
        # ax.show(block=False),
        # ax.pause(1)
        # ax.close()

    plt.tight_layout()
    plt.suptitle(video_name)
    # plt.show(block=False)
    # plt.pause(1)
    plt.savefig(os.path.join(feature_plots_output, video_name + '.png'))
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

def compute_features(timestamps, data, score):
    '''

    :param peaks_index: 4 timestamps when the hands are raised to the highest positions
    :param data: 25 body joints extracted from "GOOD" frames, which have the 10 upper body joints
    :return: features in the order of CURR_FEATURES
    '''
    features = np.empty(shape=(0, len(CURR_FEATURES)))
    for timestamp in timestamps:
        frame = data[timestamp, :]
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

        '''
        Torso:
        pt_1: neck
        pt_8: mid-hip 
        '''
        pt_1 = np.array([frame[1 * 2], frame[1 * 2 + 1]])
        pt_8 = np.array([frame[8 * 2], frame[8 * 2 + 1]])

        # vec_32: vector from pt_3 to pt_2
        vec_32 = pt_2 - pt_3
        vec_34 = pt_4 - pt_3
        left_elbow_angle = angle_between(vec_32, vec_34)

        vec_65 = pt_5 - pt_6
        vec_67 = pt_7 - pt_6
        right_elbow_angle = angle_between(vec_65, vec_67)

        # Compute the distance between left hand and right hand (pt7 and pt4)
        vec_47 = pt_7 - pt_4
        hand_distance = np.linalg.norm(vec_47)

        # Compute the shoulder distance between pt5 and pt2
        vec25 = pt_5 - pt_2
        shoulder_distance = np.linalg.norm(vec25)

        hand_shoulder_ratio = hand_distance/shoulder_distance

        # Compute midterm between hands
        hands_midpoint = (pt_4 + pt_7)/2

        # Vertical vector between pt_1 and pt_8
        vec_81 = pt_1 - pt_8
        vec_8_hand_mid = hands_midpoint - pt_8
        torso_tilted_angle = angle_between(vec_81, vec_8_hand_mid)

        # Compute the tilted angle between two hands
        horizontal_vec = np.array([pt_7[0], pt_4[1]]) - pt_4
        hand_tilted_angle = angle_between(horizontal_vec, vec_47)

        # Compute the torso area which the area enclosed by body points: 2, 5, 9, 12
        torso = np.array([pt_2, pt_5, pt_12, pt_9])
        torso_area = compute_poly_area(torso[:, 0], torso[:, 1])

        # Compute the difference of elbow angles
        elbow_angles_diff = abs(left_elbow_angle - right_elbow_angle)
        curr_features = np.asarray([left_elbow_angle, right_elbow_angle, hand_shoulder_ratio, torso_tilted_angle, hand_tilted_angle, elbow_angles_diff])

        print_string = f'timestamp: {timestamp}'
        for i, feat in enumerate(CURR_FEATURES):
            print_string += ' {0}:{1:.1f}'.format(feat, curr_features[i])
        print_string += ' score:{:.2f}'.format(score)
        print(print_string)

        features = np.vstack((features, curr_features))

    return features

def get_peak_features(should_draw_plots, should_write_features, df, feature_txt_output, feature_plots_output):
    for filepath in glob.glob(FILE_PATH + '/*.txt', recursive=True):
        # video has shape [frames x num_joints]
        data = np.loadtxt(filepath, delimiter=',')

        video_name = os.path.basename(filepath).split('.')[0]
        subject_id = '_'.join(video_name.split('_')[2:4])
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

        # Features = [num_peaks x num_features]
        score = df.loc[subject_id]['score']
        print(subject_id)
        features = compute_features(peaks_index, data, score)

        if video_name[-1] == "_":
            # Naming is inconsistent in KIMORE, some videos has an extra underscore. The extra underscore needs to be removed
            # i.e. 'CG_Expert_E_ID9_Es1_rgb_Blur_rgb271114_123334_'
            video_name = video_name[:-1]

        if should_draw_plots:
            plot_body_joints(data, peaks_index, features, video_name, feature_plots_output)

        # Save the features
        if should_write_features:
            np.savetxt(os.path.join(feature_txt_output, video_name + '.txt'), features, delimiter=',', fmt='%1.3f')

        # Add features to dataframe
        # TODO: df record the feature values computed by taking the average of 5 repetitions. NEED TO BE FIXED!
        for i, feat in enumerate(CURR_FEATURES):
            df._set_value(subject_id, feat, np.mean(features[:,i]))


def get_features_at_all_timestamps(all_timestamps_features_output):
    for filepath in glob.glob(FILE_PATH + '/*.txt', recursive=True):
        # video has shape [frames x num_joints]
        data = np.loadtxt(filepath, delimiter=',')
        num_lines = sum(1 for line in open(filepath))
        features = compute_features(np.arange(num_lines), data)

        video_name = os.path.basename(filepath).split('.')[0]
        if video_name[-1] == "_":
            # Naming is inconsistent in KIMORE, some videos has an extra underscore. The extra underscore needs to be removed
            # i.e. 'CG_Expert_E_ID9_Es1_rgb_Blur_rgb271114_123334_'
            video_name = video_name[:-1]

        print(video_name)
        # Save the features
        np.savetxt(os.path.join(all_timestamps_features_output, video_name + '.txt'), features, delimiter=',', fmt='%1.3f')


def plot_heatmap(df, corr_type, output_folder):
    # Using Pearson Correlation
    plt.figure(figsize=(11, 11))
    cor = df.corr(method=corr_type)
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.title(f"{corr_type} correlation matrix", fontsize=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(os.path.join(output_folder, '{0}_corr_{1}_features.png'.format(corr_type, len(CURR_FEATURES))))
    plt.close()

def get_Es1_features():
    should_draw_plots = False
    should_write_features = False

    output_folder = os.path.join(OUTPUT_ROOT, f'KiMoRe_skeletal_{len(CURR_FEATURES)}_features')
    try:
        os.mkdir(output_folder)
    except OSError:
        print("Creation of the directory %s failed!" % output_folder)


    # Load score_df
    '''
    To access data using index:
            score_df.loc['NE_ID16', :]
    '''
    all_score_df = pd.read_pickle('Es1_RGB_df')

    ex1_df = pd.DataFrame()
    ex1_df['score'] = all_score_df['clinical TS Ex#1']
    # Add the feature columns to dataframe
    ex1_df = pd.concat([ex1_df, pd.DataFrame(columns=CURR_FEATURES)])

    # Get feature from selected timestamps --> peaks
    # ex1_df will be updated

    feature_txt_output = feature_plots_output = ''
    if should_write_features:
        feature_txt_output = os.path.join(output_folder, 'features')
        try:
            os.mkdir(feature_txt_output)
        except OSError:
            print("Creation of the directory %s failed!" % feature_txt_output)
            assert True

    if should_draw_plots:
        feature_plots_output = os.path.join(output_folder, 'feature_plots')
        try:
            os.mkdir(feature_plots_output)
        except OSError:
            print("Creation of the directory %s failed!" % feature_plots_output)
            assert True

    get_peak_features(should_draw_plots, should_write_features, ex1_df, feature_txt_output, feature_plots_output)

    # Convert all elements to float
    ex1_df = ex1_df.astype(float)

    # Save the feature df
    ex1_df.to_csv(os.path.join(output_folder, 'Ex1_skeletal_features.csv'))
    ex1_df.to_pickle(os.path.join(output_folder, 'Ex1_skeletal_features.pkl'))

    # # Get features from all timestamps
    # all_timestamps_features_output = os.path.join(output_folder, 'feature_all_timestamps')
    # try:
    #     os.mkdir(all_timestamps_features_output)
    # except OSError:
    #     print("Creation of the directory %s failed!" % all_timestamps_features_output)
    # get_features_at_all_timestamps(all_timestamps_features_output)

    # Plot heat map for features
    plot_heatmap(ex1_df, 'pearson', output_folder)
    plot_heatmap(ex1_df, 'spearman', output_folder)

    # Create a txt file to record feature info
    with open(os.path.join(output_folder, 'features_info.txt'), 'w') as f:
        for item in CURR_FEATURES:
            f.write("%s\n" % item)

def main():
    get_Es1_features()

if __name__ == '__main__':
    main()
