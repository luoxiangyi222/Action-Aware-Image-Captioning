"""
Author: Xiangyi Luo

"""

import json
import numpy as np
from code.data_loader import DataLoader
import code.data_loader as dl

# for each corresponding OCR, we find the nearest line and get 32 words
CODE_LINE_LENGTH = 32

# loading data
data_loader = DataLoader()
data_loader.load()
action_label_dict = data_loader.actions_label_dict
action_area_dict = data_loader.actions_area_dict


def iou_at_y_direction(y1_min, y1_max, y2_min, y2_max):

    # compute intersection
    if y1_max <= y2_min or y2_max <= y1_min:
        # no inter section
        return 0
    else:
        intersection = np.min([abs(y1_min-y2_max), abs(y2_min-y1_max)])
    # compute union
    arr = np.array([y1_min, y1_max, y2_min, y2_max])
    union = np.max(arr) - np.min(arr)

    return intersection/union

# input: frame OCR, action timestamp, action region
# output: 13 * 32 tensor
# noted: 32 words as a sentence, each word is a 13-dimemsion vector

for video_num in data_loader.ALL_VIDEO_NUM_STR:
    labels = action_label_dict[video_num]

    # loop every action in current video
    action_seconds = labels[:, 0]
    for sec in action_seconds:
        ocr_path = data_loader.find_action_ocr_filename(video_num, sec)

        action_region = data_loader.find_action_region(video_num, sec)
        action_y_low = action_region[2]
        action_y_high = action_region[4]

        print(action_y_low)
        print(action_y_high)

        # loading the nearest OCR file
        # find the nearest line in ocr file with action region
        max_iou_score = 0
        max_iou_text = ''

        with open(ocr_path) as f:
            ocr_data = json.load(f)
            lines = ocr_data['lines']

            for line in lines:
                line_y_low = line['vertice']['y_min']
                line_y_high = line['vertice']['y_max']

                # compute intersection over union (IoU) along y direction
                iou_score = iou_at_y_direction(action_y_low, action_y_high, line_y_low, line_y_high)

                if iou_score > 0:
                    print(iou_score)
                    print(line['text'])
                    print(line['vertice'])
                    print("============")

                if iou_score > max_iou_score:
                    max_iou_score = iou_score
                    max_iou_text = line['text']

        print(max_iou_text)
        breakpoint()
