"""
Author: Xiangyi Luo
"""

import numpy as np
import glob


path = './../dataset/ActionNet-Dataset/Actions/8_*.txt'

# get all action net output
txt_file_list = glob.glob(path)


# first column is second, second column is action labels (labels from 1 to 13)
for file_path in txt_file_list:
    file_num = file_path.split('/')[-1][:-4]
    action_array = np.array(np.loadtxt(file_path))

    print("File Number of Output: " + str(file_num))

    # drop row without actions
    action_array = action_array[action_array[:, 1] > 0]
    print(action_array)
    breakpoint()





