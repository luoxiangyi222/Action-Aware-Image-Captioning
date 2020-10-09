"""
Author: Xiangyi Luo

"""

from code.data_loader import DataLoader
import code.data_loader as dl

# for each corresponding OCR, we find the nearest line and get 32 words
CODE_LINE_LENGTH = 32

# loading data
data_loader = DataLoader()
# data_loader.load()
action_label_dict = data_loader.actions_label_dict
action_area_dict = data_loader.actions_area_dict


# input: frame OCR, action timestamp, action region
# output: 13 * 32 tensor
# noted: 32 words as a sentence, each word is a 13-dimemsion vector

file = dl.num_to_ocr_filename('8_0', 5)



