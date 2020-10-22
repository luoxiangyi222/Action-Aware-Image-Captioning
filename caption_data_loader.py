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


class CaptionDataLoader(object):
    def __init__(self):

        self.ALL_VIDEO_ID_STR = []
        self.TOTAL_CLASS_NUM = 13  # total number of actions

        self.code_vectors = None
        self.actions_label_dict = defaultdict(lambda: {})  # only contains seconds having action
        self.actions_area_dict = {}  # contains every second
        self.action_caption_dict = defaultdict(lambda: {})
        self.action_caption_vectorized_dict = defaultdict(lambda: {})
        self.ocr_caption_timestamp_dict = {}  # record secons having ocr and subtitle

        self.row_caption_dict = defaultdict(lambda: {})

        self.formatted_ocr_action_dict = defaultdict(lambda: {})

        self.find_all_video_id()

    def load(self):
        self.load_code_vectors()
        self.load_action_one_hot()
        self.load_action_region()
        self.load_ocr_caption_timestamp()
        self.load_row_caption()
        self.load_formatted_ocr_action()
        self.load_action_caption()

    def find_action_ocr_filename(self, video_num, second):
        """
        Given the video_id and the second of the action, find the nearest OCR filename.
        :param video_num: str
        :param second: int
        :return: a filename, the related ocr file
        """
        ocr_timestamps = self.ocr_caption_timestamp_dict[video_num]

        nearest_ocr_sec = find_nearest(ocr_timestamps, second)

        ocr_name = '0' * (5 - len(str(nearest_ocr_sec))) + str(nearest_ocr_sec)

        ocr_prefix = './../dataset/OCR/'
        ocr_file = ocr_prefix + video_num + '/' + ocr_name + '.json'

        return ocr_file

    def find_all_video_id(self):
        """
        Find all video_id
        :return:
        """
        path = './../dataset/Actions/8_*.txt'
        txt_file_list = glob.glob(path)
        for file_path in txt_file_list:
            file_num_str = file_path.split('/')[-1][:-4]
            self.ALL_VIDEO_ID_STR.append(file_num_str)

    def code_token_to_vector(self, token):
        """
        Find the vector representation of the given code token
        :param token: string, code token
        :return: token vector
        """
        return self.code_vectors[token]

    def find_action_region(self, video_num, second):
        """
        Given the video id and second, find the action interaction area
        :param video_num: string, video id
        :param second: timestamp in that video
        :return: A entry of the array to tell the action region
        """
        second = int(second)
        return self.actions_area_dict[video_num][second-1]

    def find_action_caption(self, video_id, action_sec):
        """
        Given the video id and second, find the nearest caption associated with the action
        :param video_id: string, video id
        :param action_sec: timestamp in that video
        :return: the corresponding caption of the action
        """
        timestamp = self.ocr_caption_timestamp_dict[video_id]
        action_sec = int(action_sec)
        nearest_caption_sec = find_nearest(timestamp, action_sec)
        return self.row_caption_dict[video_id][nearest_caption_sec]

    def load_ocr_caption_timestamp(self):
        """
        For each video, not every second have OCR file and caption line.
        So, find and record all end timestamps for each video having OCR file and Captions.
        :return:
        """
        for video_number in self.ALL_VIDEO_ID_STR:
            filepath = './../dataset/Captions/' + video_number + '.txt'

            # recording all end timestamps where having ocr and subtitle
            caption_data = pd.read_csv(filepath, sep=' ', header=None, usecols=[0, 1])
            caption_data.columns = ['start', 'end']
            timestamps = caption_data[['end']]
            timestamps = np.array(timestamps.unstack())

            # record timestamp having ocr and subtitle
            self.ocr_caption_timestamp_dict[video_number] = timestamps

    def load_row_caption(self):
        """
        Just loading all captions.
        :return: Update self.row_caption_dict, the keys of dictionary is not related to actions.
        """
        for video_number in self.ALL_VIDEO_ID_STR:
            filepath = './../dataset/Captions/' + video_number + '.txt'
            # loading subtitles
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    words = line.split(' ')
                    end_sec = int(words[1])
                    subtitle = ' '.join(words[2:-1])
                    self.row_caption_dict[video_number][end_sec] = subtitle

    def load_action_caption(self):
        """
        This should be more useful when training than raw caption dict.
        Computing captions with all actions.
        :return: Update self.action_caption_dict, the keys of dictionary is based on actions.
        """
        for video_num in self.ALL_VIDEO_ID_STR:

            # find every action in current video
            action_seconds = self.actions_label_dict[video_num].keys()

            for action_sec in action_seconds:
                # find nearest caption with action
                action_caption = self.find_action_caption(video_num, action_sec)
                self.action_caption_dict[video_num][action_sec] = action_caption

    def update_action_caption_vectorized_dict(self, video_id, action_sec, sentence_list):
        """
        :return:
        """
        self.action_caption_vectorized_dict[video_id][action_sec] = sentence_list

    def load_formatted_ocr_action(self):
        """
        Load the tensor for each data. One tensor contains the 32 words and the action info.
        :return:
        """
        prefix = './transformer_input/'
        for video_num in self.ALL_VIDEO_ID_STR:
            caption_dict = np.load(prefix + video_num + '.npy', allow_pickle=True)
            self.formatted_ocr_action_dict[video_num] = caption_dict[()]

    def load_code_vectors(self):
        """
        Loading the parameters of the pre_trained Word2Vec model.
        :return:
        """
        # Loading from saved word embeddings
        self.code_vectors = KeyedVectors.load("word2vec.model").wv

    def load_action_one_hot(self):
        """
        Load action labels converted to one-hot format.
        :return:
        """
        path = './../dataset/Actions/8_*.txt'

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

            for one_hot in one_hot_array:
                sec = int(one_hot[0])
                encoding = one_hot[1:].astype('int32')
                self.actions_label_dict[file_num_str][sec] = encoding

            # self.actions_label_dict[file_num_str] = one_hot_array

    def load_action_region(self):
        """
        Load the file listing all action interaction regions.
        :return:
        """

        txt_path = './../dataset/Annotations/8_*.txt'
        txt_file_list = glob.glob(txt_path)

        for txt_file in txt_file_list:
            file_num_str = txt_file.split('/')[-1][:-4]
            array = np.array(np.loadtxt(txt_file))
            array = array[:, :-2]
            self.actions_area_dict[file_num_str] = array



# testing code
# ddd = DataLoader()
# ddd.load()
# print(len(ddd.ocr_subtitle_timestamp_dict))
# print(ddd.ocr_subtitle_timestamp_dict)
# print(ddd.subtitle_dict['8_0'])

# print(ddd.actions_label_dict['8_0'])
# print(len(ddd.action_caption_vectorized_dict_vectorized_dict))
