import glob
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


# The difference between the y-coordinates of left-hand and right-hand
#  should not exceed HANDS_DIFF_THRESHOLD
HANDS_DIFF_THRESHOLD = 50



def show_plots(frame):
    xs = []
    ys = []
    for i in range(0, len(frame)):
        if i % 2:
            ys.append(frame[i])
        else:
            xs.append(frame[i])
    plt.plot(xs, ys, 'bo')

    index = 0
    for x, y in zip(xs, ys):
        if (index == 9): index = 11
        label = "{}".format(index)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        index += 1
    plt.gca().invert_yaxis()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def get_peaks_index(data, num_peaks):

    '''
    From testing, if there is less than 64 frames, then split data into 4 sections won't work well.
    The local peaks found were not accurate.

    '''

    if (data.shape[0] < 5): return np.argmax(data)

    # split into data into num_peaks number of sections and find 1 peak in each section
    split_data = np.array_split(data, num_peaks)

    index_count = 0
    peaks_index = []
    for array in split_data:
        ind = np.argmax(array)
        peaks_index.append(ind + index_count)
        index_count += len(array)

    return peaks_index


def main():
    FILE_PATH = '/Users/Clara_1/Documents/University/Year4/Thesis/Code/pytorch-openpose/my_data/output_3_12'
    for filepath in glob.glob(FILE_PATH + '/*.txt', recursive=True):
        filepath = FILE_PATH + "/GPP_Stroke_S_ID6_Es1_rgb_Blur_rgb040716_112230.txt"

        # video has shape [frames x num_joints]
        video = np.loadtxt(filepath, delimiter=',')

        "rgb130616_110835"
        # show the point plot
        # show_plots(video[0])

        # left hand : point 4
        left_hand_x = video[:, 4 * 2]
        left_hand_y = video[:, 4 * 2 + 1]

        # right hand : point 7
        right_hand_x = video[:, 7 * 2]
        right_hand_y = video[:, 7 * 2 + 1]

        # Subjects were asked to repeat each exercise consecutively 5 times
        # Find 5 local min for y-coordinates (local min == hand rise above head)

        sum_left_right = left_hand_y + right_hand_y
        sum_left_right *= -1
        diff_left_right = abs(left_hand_y - right_hand_y)

        peaks_index = get_peaks_index(sum_left_right, 4)  # Find 4 peaks
        print(peaks_index)

        for index in peaks_index:
            frame = video[index, :]
            show_plots(frame)

        print(1)



if __name__ == '__main__':
    main()
