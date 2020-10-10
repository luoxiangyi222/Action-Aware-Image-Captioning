"""
Author: Xiangyi Luo

"""
import json
import numpy as np
from code.data_loader import DataLoader
from code.code_data_preprocessing import CodePreprocessor

code_pre = CodePreprocessor()

# for each corresponding OCR, we find the nearest line and get 32 words
CODE_LINE_LENGTH = 32

# loading data
data_loader = DataLoader()
data_loader.load()


def iou_at_y_direction(y1_min, y1_max, y2_min, y2_max):

    # compute intersection
    if y1_max <= y2_min or y2_max <= y1_min:
        # no inter section
        return 0
    elif y1_max > y2_max and y1_min < y2_min:
        return abs(y2_max-y2_min)
    elif y2_max > y1_max and y2_min < y1_min:
        return abs(y1_max - y1_min)
    else:
        intersection = np.min([abs(y1_min-y2_max), abs(y2_min-y1_max)])
    # compute union
    arr = np.array([y1_min, y1_max, y2_min, y2_max])
    union = np.max(arr) - np.min(arr)

    return intersection/union


def find_action_relevant_words(ocr_file, action_y_min, action_y_max, word_count):
    """

    :param ocr_file: OCR files
    :param action_y_min: int, action y lower
    :param action_y_max: int, action y higher
    :param word_count: how many words are wanted for relevant actions
    :return: a list of word vectors represent the relevant words
    """
    lines = ocr_file['lines']

    # find the most relevant line
    most_relevant_line = None
    max_iou_score = 0

    for line in lines:

        line_y_low = line['vertice']['y_min']
        line_y_high = line['vertice']['y_max']

        print(action_y_low)
        print(action_y_high)
        print(line_y_low)
        print(line_y_high)
        print('----------')

        # compute intersection over union (IoU) along y direction
        iou_score = iou_at_y_direction(action_y_min, action_y_max, line_y_low, line_y_high)

        if iou_score > max_iou_score:

            max_iou_score = iou_score
            most_relevant_line = line

    # if no ocr line is relative
    if most_relevant_line is None:
        most_relevant_y_min = action_y_low
    else:
        most_relevant_y_min = most_relevant_line['vertice']['y_min']

    # sort lines by y_min difference
    sorted_lines = sorted(lines, key=lambda k: abs(k['vertice']['y_min'] - most_relevant_y_min))
    sorted_lines = [line['text'] for line in sorted_lines]

    # extract 32 words
    selected_words = []
    for line in sorted_lines:
        pre_line = code_pre.__call__([line])
        selected_words.extend(pre_line[0])
        if len(selected_words) >= word_count:
            break

    selected_words = selected_words[:word_count]

    # convert to vectors
    selected_word_vectors = []
    for word in selected_words:
        try:
            wv = data_loader.word_to_vector(word)
            selected_word_vectors.append(wv)

        except KeyError:
            # if not found, fit in zeros
            selected_word_vectors.append(np.zeros((data_loader.TOTAL_CLASS_NUM,)))
            error_message = 'word ' + word + ' not in vocabulary'
            print('Error: ' + error_message)

    selected_word_vectors = np.array(selected_word_vectors)

    return selected_word_vectors


# input: frame OCR, action timestamp, action region
# output: 32*13 tensor
# noted: 32 words as a sentence, each word is a 13 dimension vector
for video_num in data_loader.ALL_VIDEO_NUM_STR:

    labels = data_loader.actions_label_dict[video_num]

    # find every action in current video
    action_seconds = labels[:, 0]

    for action_sec in action_seconds:
        # find nearest ocr file with action
        ocr_path = data_loader.find_action_ocr_filename(video_num, action_sec)

        # find action interaction region
        action_region = data_loader.find_action_region(video_num, action_sec)
        action_y_low = action_region[2]
        action_y_high = action_region[4]
        # find action label
        # TODO


        # extract most relevant 32 words in ocr file
        with open(ocr_path) as f:
            ocr_data = json.load(f)
            # 32 x 13 tensor
            relevant_words_tensor = find_action_relevant_words(ocr_data, action_y_low, action_y_high, 32)


        # print info
        print(video_num)
        print(action_sec)
        print(action_y_low)
        print(action_y_high)
        print('===================')


