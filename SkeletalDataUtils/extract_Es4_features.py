import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import math
# Turn interactive plotting off
plt.ioff()

CURR_FEATURES = ['']
NUM_REPETITION = 5
EXERCISE_INDEX = 4
EXERCISE_TYPE = f'Es{EXERCISE_INDEX}'

FILE_PATH = f'/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/{EXERCISE_TYPE}/KiMoRe_skeletal_txt_files_all_joints'
OUTPUT_ROOT = f'/Users/Clara_1/Documents/University/Year4/Thesis/Datasets/KiMoRe/{EXERCISE_TYPE}'

# CURR_FEATURES = ['torso_tilted_angle', 'left_knee_angle', 'right_knee_angle']
CURR_FEATURES = ['torso_tilted_angle', 'knee_dist_ratio', 'shoulder_level_angle']

NUM_PEAKS = NUM_REPETITION*2

def plot_body_joints(data, peaks_index, features, video_name, feature_plots_output):
    fig = plt.figure(figsize=(15, 15))

    for index, counter in zip(peaks_index, range(NUM_PEAKS)):
        ax = fig.add_subplot(3, 4, counter + 1)
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
        ax.plot(non_zero_xs, non_zero_ys, 'bo', markersize=8)
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
        # for i, feat in enumerate(CURR_FEATURES):
            # title += ' {0}:{1:.1f}'.format(feat, feature[i])

        ax.set_title(title, fontsize=17)
        # ax.show(block=False),
        # ax.pause(1)
        # ax.close()

    plt.suptitle(video_name, fontsize=20)
    # plt.show(block=False)
    # plt.pause(1)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(feature_plots_output, video_name + '.png'))
    plt.close()

def get_peaks_index(data_df, selected_data, num_peaks):

    '''
    From testing, if there is less than 64 frames, then split data into 5 sections won't work well.
    The local peaks found won't not accurate.

    '''

    # split into data into num_peaks number of sections and find 1 peak in each section
    # split_data = np.array_split(selected_data, num_peaks)
    split_data = np.array_split(data_df, num_peaks)

    index_count = 0
    peaks_index = []
    for df in split_data:

        # Subjects were asked to repeat each exercise consecutively 5 times
        # Find 5 local min for y-coordinates (local min == hand rise above head)
        left_most = df['x_sum'].idxmin()
        right_most = df['x_sum'].idxmax()


        peaks_index.append(left_most)
        peaks_index.append(right_most)


    # Should be a left peak and right peak
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
        Hip:
        pt_9: left hip
        pt_12: right hip
        
        pt_10: left knee
        pt_13: right knee
        
        pt_2: left shoulder
        pt_5: right shoulder
        '''
        pt_9 = np.array([frame[9 * 2], frame[9 * 2 + 1]])
        pt_12 = np.array([frame[12 * 2], frame[12 * 2 + 1]])

        pt_10 = np.array([frame[10 * 2], frame[10 * 2 + 1]])
        pt_13 = np.array([frame[13 * 2], frame[13 * 2 + 1]])

        pt_2 = np.array([frame[2 * 2], frame[2 * 2 + 1]])
        pt_5 = np.array([frame[5 * 2], frame[5 * 2 + 1]])
        '''
        Torso:
        pt_1: neck
        pt_8: mid-hip 
        '''
        pt_1 = np.array([frame[1 * 2], frame[1 * 2 + 1]])
        pt_8 = np.array([frame[8 * 2], frame[8 * 2 + 1]])

        # # vec_32: vector from pt_3 to pt_2
        # vec_10_9 = pt_9 - pt_10
        # vec_10_11 = pt_4 - pt_3
        # left_elbow_angle = angle_between(vec_32, vec_34)
        #
        # vec_65 = pt_5 - pt_6
        # vec_67 = pt_7 - pt_6
        # right_elbow_angle = angle_between(vec_65, vec_67)
        #
        # # Compute shoulder angles
        # vec21 = pt_1 - pt_2
        # vec23 = pt_3 - pt_2
        # left_shoulder_angle = angle_between(vec21, vec23)
        #
        # vec51 = pt_1 - pt_5
        # vec56 = pt_6 - pt_5
        # right_shoulder_angle = angle_between(vec51, vec56)
        #
        #
        # Compute the distance between left hip and right hip (pt9 and pt12)
        vec_9_12 = pt_9 - pt_12
        hip_distance = np.linalg.norm(vec_9_12)

        # Compute the knee distance between pt10 and pt13
        vec_10_13 = pt_10 - pt_13
        knee_distance = np.linalg.norm(vec_10_13)
        knee_dist_ratio = hip_distance/knee_distance

        # Vertical vector between pt_1 and pt_8
        vec_81 = pt_1 - pt_8

        # Compute vertical vector that goes through pt_8
        vec_8_vertical = np.array([pt_8[0], pt_1[1]]) - pt_8
        torso_tilted_angle = angle_between(vec_81, vec_8_vertical)

        # Compute the tilted angle between two shoulders
        vec_25 = pt_5 - pt_2
        vec_2_horizontal = np.array([pt_5[0], pt_2[1]]) - pt_2
        shoulder_level_angle = angle_between(vec_25, vec_2_horizontal)
        #
        # # Compute the torso area which the area enclosed by body points: 2, 5, 9, 12
        # torso = np.array([pt_2, pt_5, pt_12, pt_9])
        # torso_area = compute_poly_area(torso[:, 0], torso[:, 1])
        #
        # # Compute the difference of elbow angles
        # elbow_angles_diff = abs(left_elbow_angle - right_elbow_angle)
        #
        # # Compute angle between shoulder-shoulder and hand-hand
        # shoulders_hands_angle = angle_between(vec_47, vec25)
        #
        # # Compute angle between arm and torso
        # vec_18 = pt_8 - pt_1
        # vec_14 = pt_4 - pt_1
        # vec_17 = pt_7 - pt_1
        #
        # left_arm_torso_angle = angle_between(vec_14, vec_18)
        # right_arm_torso_angle = angle_between(vec_17, vec_18)

        # Append all features to curr_features
        curr_features = np.asarray([torso_tilted_angle, knee_dist_ratio, shoulder_level_angle])

        assert curr_features.size == len(CURR_FEATURES)
        print_string = f'timestamp: {timestamp}'
        for i, feat in enumerate(CURR_FEATURES):
            print_string += ' {0}:{1:.1f}'.format(feat, curr_features[i])
        print_string += ' score:{:.2f}'.format(score)
        print(print_string)


        features = np.vstack((features, curr_features))

    # add 1 additional row [max(torso_tilted_angle)-min(torso_tilted_angle), mean(torso_tilted_angle)]
    # additional_feat = [max(features[:,0]) - min(features[:,0]), np.mean(features[:,0])]
    # print('additional row: {0}:{1:.1f}  {2}:{3:.1f}'.format('min/max', additional_feat[0], 'mean:', additional_feat[1]))
    # features = np.vstack((features, additional_feat))
    return features

def compute_distance_btw_2_points(x1, y1, x2, y2):
    return pow((pow(x1 - x2, 2) +
                pow(y1 - y2, 2))
             , 0.5)

def get_peak_features(should_draw_plots, should_write_features, df, feature_txt_output, feature_plots_output):
    for filepath in glob.glob(FILE_PATH + '/*.txt', recursive=True):
        # video has shape [frames x num_joints]
        data = np.loadtxt(filepath, delimiter=',')

        # isValid = 'E_ID12' in filepath or 'P_ID11' in filepath
        # if not isValid: continue

        video_name = os.path.basename(filepath).split('.')[0]
        subject_id = '_'.join(video_name.split('_')[2:4])
        print(subject_id)

        # show the point plot
        # show_plots(data[0])

        # left hip : point 9
        left_hip_x = data[:, 9 * 2]
        left_hip_y = data[:, 9 * 2 + 1]

        # right hip : point 12
        right_hip_x = data[:, 12 * 2]
        right_hip_y = data[:, 12 * 2 + 1]

        # Subjects were asked to repeat each exercise consecutively 5 times
        # Find 5 local min for y-coordinates (local min == hand rise above head)

        sum_left_right = left_hip_x + right_hip_x

        d = { 'x_sum': sum_left_right }

        data_df = pd.DataFrame(data=d)


        peaks_index = get_peaks_index(data_df, sum_left_right, NUM_REPETITION)  # Find 5 peaks, each has LEFT & RIGHT peak

        # Features = [num_peaks x num_features]
        score = df.loc[subject_id]['TS']
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
    sns.heatmap(cor, annot=True, cmap='coolwarm')
    plt.title(f"{corr_type} correlation matrix", fontsize=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(os.path.join(output_folder, '{0}_corr_{1}_features.png'.format(corr_type, len(CURR_FEATURES))))
    plt.close()

def get_features():
    should_draw_plots = True
    should_write_features = True

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
    all_score_df = pd.read_pickle(f'{EXERCISE_TYPE}_RGB_df')

    ex_df = pd.DataFrame()
    ex_df['TS'] = all_score_df[f'clinical TS Ex#{EXERCISE_INDEX}']
    # ex_df['PO'] = all_score_df[f'clinical PO Ex#{EXERCISE_INDEX}']
    # ex_df['CF'] = all_score_df[f'clinical CF Ex#{EXERCISE_INDEX}']

    # score_df = ex_df.sort_values(by=['score'])

    # Add the feature columns to dataframe
    ex_df = pd.concat([ex_df, pd.DataFrame(columns=CURR_FEATURES)])

    # Get feature from selected timestamps --> peaks
    # ex_df will be updated
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

    get_peak_features(should_draw_plots, should_write_features, ex_df, feature_txt_output, feature_plots_output)

    # Convert all elements to float
    ex_df = ex_df.astype(float)

    # Save the feature df
    ex_df.to_csv(os.path.join(output_folder, f'Ex{EXERCISE_INDEX}_skeletal_features.csv'))
    ex_df.to_pickle(os.path.join(output_folder, F'Ex{EXERCISE_INDEX}_skeletal_features.pkl'))

    # # Get features from all timestamps
    # all_timestamps_features_output = os.path.join(output_folder, 'feature_all_timestamps')
    # try:
    #     os.mkdir(all_timestamps_features_output)
    # except OSError:
    #     print("Creation of the directory %s failed!" % all_timestamps_features_output)
    # get_features_at_all_timestamps(all_timestamps_features_output)

    # Plot heat map for features
    plot_heatmap(ex_df, 'pearson', output_folder)
    # plot_heatmap(ex_df, 'spearman', output_folder)

    # Create a txt file to record feature info
    with open(os.path.join(output_folder, 'features_info.txt'), 'w') as f:
        for item in CURR_FEATURES:
            f.write("%s\n" % item)

def main():
    get_features()

if __name__ == '__main__':
    main()
