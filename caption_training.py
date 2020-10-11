"""
Author: xiangyi
Tutorial:
https://www.tensorflow.org/tutorials/text/image_captioning

This model convert cations to int.

"""

import tensorflow as tf
import time
from data_loader import DataLoader
import caption_model as cp_model
import matplotlib.pyplot as plt
import numpy as np


def calc_max_length(list_of_list_word):
    return max(len(t) for t in list_of_list_word)


# ##################### Preprocess and tokenize the captions ####################
data_loader = DataLoader()
data_loader.load()
captions = data_loader.row_caption_dict.copy()
# print(captions)

train_captions = []
for video_num, v_dict in data_loader.action_caption_dict.items():

    lines = v_dict.values()
    lines = ['<start> ' + line + ' <end>' for line in lines]
    lines = [line.split(' ') for line in lines]
    train_captions.extend(lines)

# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

# convert action related caption to vector
i = 0
for v_id, v_dict in data_loader.action_caption_dict.items():
    keys = v_dict.keys()

    for k in keys:
        data_loader.update_action_caption_vectorized_dict(v_id, k, cap_vector[i])
        i = i + 1

# ######################## Dataset #####################################
# X = data_loader.formatted_ocr_action_dict
# y = data_loader.action_caption_vectorized_dict

num_samples = int(len(cap_vector) / 10 * 8)

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = num_samples  # BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 13
attention_features_shape = 33

X = []
for v_id, ocr_act_dict in data_loader.formatted_ocr_action_dict.items():
    X.extend(list(ocr_act_dict.values()))
X = np.array(X)

Y = []
for v_id, target_dict in data_loader.action_caption_vectorized_dict.items():
    Y.extend(list(target_dict.values()))
Y = np.array(Y)

#  splitting into training and testing data
index = np.array(range(len(X)))
np.random.shuffle(index)
divide_at = int(len(X)/10*8)
X = X[index]
Y = Y[index]

train_X = tf.convert_to_tensor(X[: divide_at])
train_Y = tf.convert_to_tensor(Y[: divide_at])

val_X = tf.convert_to_tensor(X[divide_at:])
val_Y = tf.convert_to_tensor(Y[divide_at:])

train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ######################## model #####################################


encoder = cp_model.CNN_Encoder(embedding_dim)
decoder = cp_model.RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# ######################## checkpoint #####################################
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

# ######################## training #####################################
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


EPOCHS = 20
num_steps = len(train_X)  # how many images?

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(train_dataset):

        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss / num_steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# plt.plot(loss_plot)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Plot')
# plt.show()

print('Training finished.')


# ############### Evaluation ###############

def evaluate(image):

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    # temp_input = tf.expand_dims(load_image(image)[0], 0)
    # img_tensor_val = image_features_extract_model(temp_input)
    # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(image)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


real_file_path = 'real_caption.txt'
pred_file_path = 'pred_caption.txt'

real_f = open(real_file_path, 'w')
pred_f = open(pred_file_path, 'w')

# captions on the validation set
for i, val_x in enumerate(val_X):

    val_y = np.array(tf.gather(val_Y, i))
    real_caption = ' '.join([tokenizer.index_word[j] for j in val_y if j not in [0]])
    result, _ = evaluate(val_x)
    pred_caption = ' '.join(result)
    real_f.write(real_caption)
    pred_f.write(pred_caption)
    print('//////////////////////////////////////////////////')
    print('Real Caption:', real_caption)
    print('Prediction Caption:', pred_caption)

real_f.close()
pred_f.close()
