"""
Author: Xiangyi Luo

This module load necessary data for future analysis.

"""
import numpy as np
import glob
import pandas as pd
from gensim.models import KeyedVectors
from collections import defaultdict


OCR_FILE_NAME_LENGTH = 5  # OCR file length


def num_to_one_hot(label, total_classes):
    """
    :param label: numerical value, label the class of the action, note the label start from 1, 0 means nothing
    :param total_classes: the total number of all classes, also the length of output encoding
    :return: 1d array, one hot encoding for that label
    """
    if label > 0:
        one_hot_dode = np.zeros(total_classes,)
        label = int(label)
        one_hot_dode[label-1] = 1
        return one_hot_dode
    else:
        return None


def find_filename_action_region(video_num):
    annotation_file = './../dataset/Annotations/' + video_num + '.txt'
    return annotation_file


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class DataLoader(object):
    def __init__(self):

        self.ALL_VIDEO_NUM_STR = []
        self.TOTAL_CLASS_NUM = 13  # total number of actions

        self.code_vectors = None
        self.actions_label_dict = {}  # only contains seconds having action
        self.actions_area_dict = {}  # contains every second
        self.ocr_subtitle_timestamp_dict = {}  # record secons having ocr and subtitle

        self.subtitle_dict = defaultdict(lambda: {})

        self.find_all_video_num()

    def load(self):
        self.load_code_vectors()

        self.load_action_one_hot()
        self.load_action_region()
        self.load_ocr_subtitle_timestamp()

    def find_action_ocr_filename(self, video_num, second):
        """
        :param video_num: str
        :param second: int
        :return: a filename, the related ocr file
        """
        ocr_timestamps = self.ocr_subtitle_timestamp_dict[video_num]

        nearest_ocr_sec = find_nearest(ocr_timestamps, second)

        ocr_name = '0' * (5 - len(str(nearest_ocr_sec))) + str(nearest_ocr_sec)

        ocr_prefix = './../dataset/OCR/'
        ocr_file = ocr_prefix + video_num + '/' + ocr_name + '.json'

        return ocr_file

    def find_all_video_num(self):
        path = './../dataset/ActionNet-Dataset/Actions/8_*.txt'
        txt_file_list = glob.glob(path)
        for file_path in txt_file_list:
            file_num_str = file_path.split('/')[-1][:-4]
            self.ALL_VIDEO_NUM_STR.append(file_num_str)

    def word_to_vector(self, word):
        return self.code_vectors[word]

    def find_action_region(self, video_num, second):
        second = int(second)
        return self.actions_area_dict[video_num][second-1]

    def find_action_subtitle(self, video_num, second):
        second = int(second)
        return self.subtitle_dict[video_num][second-1]

    def load_ocr_subtitle_timestamp(self):
        for video_number in self.ALL_VIDEO_NUM_STR:

            filepath = './../dataset/Caption/' + video_number + '.txt'

            # recording all end timestamps where having ocr and subtitle
            caption_data = pd.read_csv(filepath, sep=' ', header=None, usecols=[0, 1])
            caption_data.columns = ['start', 'end']
            timestamps = caption_data[['end']]
            timestamps = np.array(timestamps.unstack())
            # record timestamp having ocr and subtitle
            self.ocr_subtitle_timestamp_dict[video_number] = timestamps

            # loading subtitles
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    words = line.split(' ')
                    end_sec = int(words[1])
                    subtitle = ' '.join(words[2:-1])
                    self.subtitle_dict[video_number][end_sec] = subtitle

    def load_code_vectors(self):
        # Loading from saved word embeddings
        self.code_vectors = KeyedVectors.load("word2vec.model").wv

    def load_action_one_hot(self):
        path = './../dataset/ActionNet-Dataset/Actions/8_*.txt'

        # get all action net output
        txt_file_list = glob.glob(path)

        # convert label to one-hot encoding
        # first column is second, second column is action labels (labels from 1 to 13)
        for file_path in txt_file_list:
            file_num_str = file_path.split('/')[-1][:-4]
            action_array = np.array(np.loadtxt(file_path))

            # drop rows without actions
            action_array = action_array[action_array[:, 1] > 0]

            # convert to one hot code
            sec_col = action_array[:, 0]
            lab_col = action_array[:, 1]

            one_hot_array = []
            for lab in lab_col:
                one_hot_array.append(num_to_one_hot(lab, self.TOTAL_CLASS_NUM))

            one_hot_array = np.hstack([sec_col.reshape(len(sec_col), 1), one_hot_array])
            self.actions_label_dict[file_num_str] = one_hot_array

    def load_action_region(self):

        txt_path = './../dataset/ActionNet-Dataset/Annotations/8_*.txt'
        txt_file_list = glob.glob(txt_path)

        for txt_file in txt_file_list:
            file_num_str = txt_file.split('/')[-1][:-4]
            array = np.array(np.loadtxt(txt_file))
            array = array[:, :-2]
            self.actions_area_dict[file_num_str] = array


# testing code
ddd = DataLoader()
ddd.load()
# print(len(ddd.ocr_subtitle_timestamp_dict))
# print(ddd.ocr_subtitle_timestamp_dict)
# print(ddd.subtitle_dict['8_0'])