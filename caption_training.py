"""
Author: Xiangyi Luo
"""

# load input tensor and cap vector
import numpy as np
from code.data_loader import DataLoader
data_loader = DataLoader()
data_loader.load()

# get input
input_tensor = []

for _, v_dict in data_loader.formatted_ocr_action_dict.items():
    tensor_list = (list(v_dict.values()))
    input_tensor.extend(tensor_list)
input_tensor = np.array(input_tensor)
print(input_tensor.shape)

# get target
target_tensor = []
for v_num, v_dict in data_loader.action_caption_vectorized_dict.items():
    tensor_list = v_dict.values()
    target_tensor.extend(tensor_list)
target_tensor = np.array(target_tensor)

print(target_tensor.shape)
print(target_tensor[0])

# ########### create tf.data dataset for training
# Feel free to change these parameters according to your system's configuration

# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# embedding_dim = 256
# units = 512
# top_k = 5000
# vocab_size = top_k + 1
# num_steps = len(img_name_train)  # BATCH_SIZE
# # Shape of the vector extracted from InceptionV3 is (64, 2048)
# # These two variables represent that vector shape
# features_shape = 2048
# attention_features_shape = 64