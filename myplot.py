
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2

"""
no cnn
no shuffle dataset
epoch 50
"""
# loss = [2.5438898, 2.0961719, 1.9342057, 1.8309572, 1.7562113, 1.6951047, 1.6404074, 1.5895994, 1.5387768, 1.4872735, 1.4360892, 1.3866911, 1.3364153, 1.2863322, 1.2343456, 1.1862146, 1.1399753, 1.093351, 1.0528772, 1.0076101, 0.9648413, 0.929225, 0.8905379, 0.8526682, 0.8194706, 0.79123867, 0.7589013, 0.7318305, 0.7048246, 0.67858607, 0.65290976, 0.6303215, 0.60679835, 0.5836884, 0.57099265, 0.54749125, 0.5238278, 0.51135737, 0.4965724, 0.47805226, 0.46161464, 0.4508476, 0.43205553, 0.42687896, 0.41569978, 0.4055779, 0.38989863, 0.3728068, 0.3613119, 0.34883776]
# plt.figure()
# plt.title('Transformer training loss: Epoch 50')
# plt.plot(loss)
# plt.show()


img = cv2.imread('00163.jpg')
img = cv2.rectangle(img, (234, 300), (1054, 720), (255, 0, 0), thickness=2)

# img = cv2.rectangle(img, (365, 0), (587, 29), (0, 0, 255), thickness=2)
# img = cv2.rectangle(img, (733, 0), (885, 29), (0, 0, 255), thickness=2)
# img = cv2.rectangle(img, (1031, 0), (1167, 29), (0, 0, 255), thickness=2)
#
# img = cv2.rectangle(img, (77, 1), (222, 29), (0, 0, 255), thickness=2)
# img = cv2.rectangle(img, (106, 52), (721, 87), (0, 0, 255), thickness=2)
# img = cv2.rectangle(img, (107, 137), (975, 169), (0, 0, 255), thickness=2)
#
# img = cv2.rectangle(img, (104, 220), (999, 255), (0, 0, 255), thickness=2)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

