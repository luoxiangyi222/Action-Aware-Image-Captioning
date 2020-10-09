"""
Author: Xiangyi Luo

Processing with the action net results.

Keep rows having
"""

import numpy as np
import glob

from sklearn.preprocessing import OneHotEncoder


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

TOTAL_CLASS_NUM = 13

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

path = './../dataset/ActionNet-Dataset/Actions/8_*.txt'

# get all action net output
txt_file_list = glob.glob(path)

file_to_coding_dict = {}  # a dict of dict, the internal dict stores one hot information for each file

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

    action_array = np.hstack([sec_col.reshape(len(action_array), 1), np.array(one_hot_array)])
    action_array = action_array.astype('int32')
    np.save('./../dataset/one_hot_action/'+file_num_str, action_array)
    file_to_coding_dict[file_num_str] = action_array

print(file_to_coding_dict)






