"""
Author: Xiangyi Luo

This module load necessary data for future analysis.

"""
import numpy as np
import glob
from gensim.models import KeyedVectors


TOTAL_CLASS_NUM = 13  # total number of actions
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


def num_to_ocr_filename(video_num, second):
    """
    Given the video number, and a number indicating seconds, finds corresponding ocr file.
    This function is used to find the corresponding OCR file when there is an action occurs.
    :param video_num: video number, for example: '8_12'
    :param second: timestamp for the action
    :return: the corresponding json filename
    """
    prefix = './../dataset/OCR/'

    folder_name = str(video_num) + '/'

    filename = str(second)
    zero_num = OCR_FILE_NAME_LENGTH - len(filename)
    filename = zero_num * '0' + filename + '.json'

    filename = prefix + folder_name + filename
    return filename


class DataLoader(object):
    def __init__(self):
        self.code_vectors = None
        self.actions_label_dict = {}
        self.actions_area_dict = {}

    def load(self):
        self.load_code_vectors()
        self.load_action_one_hot()
        self.load_action_region()

    def load_code_vectors(self):
        # Loading from saved word embeddings
        self.code_vectors = KeyedVectors.load("wordvectors.kv", mmap='r')

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
                one_hot_array.append(num_to_one_hot(lab, TOTAL_CLASS_NUM))

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
# ddd = DataLoader()
# ddd.load_action_one_hot()
