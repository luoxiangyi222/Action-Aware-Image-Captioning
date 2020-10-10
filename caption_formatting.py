"""
Author: xiangyi
Tutorial:
https://www.tensorflow.org/tutorials/text/image_captioning

This model convert cations to int.

"""

import tensorflow as tf
import numpy as np
from code.data_loader import DataLoader
from collections import defaultdict


def calc_max_length(list_of_list_word):
    return max(len(t) for t in list_of_list_word)


# Preprocess and tokenize the captions
data_loader = DataLoader()
data_loader.load()
captions = data_loader.row_caption_dict.copy()
# print(captions)

train_captions = []
for video_num, v_dict in captions.items():
    # print(video_num)
    lines = v_dict.values()
    lines = [line.split(' ') for line in lines]
    train_captions.extend(lines)

print(len(train_captions))
breakpoint()

# Choose the top 5000 words from the vocabulary
top_k = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
max_length = calc_max_length(train_seqs)
#
# print(cap_vector)

save_path = './../dataset/transformer_target/'
for video_num, v_dict in captions.items():

    save_dict = {}
    keys = list(v_dict.keys())
    for i in range(len(keys)):
        sentence_vec = cap_vector[i]
        save_dict[keys[i]] = sentence_vec

    dict_len = len(v_dict)
    cap_vector = cap_vector[dict_len:]
    save_file_path = save_path + video_num + '.npy'
    np.save(save_file_path, save_dict)
    print('/////////')
    print(video_num)
    print(save_dict)

