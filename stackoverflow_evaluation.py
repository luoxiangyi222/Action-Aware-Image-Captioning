"""
Author: Xiangyi
"""
import glob
import pandas as pd
import numpy as np


def index_to_rank(i:int):
    return i+1


def top_5_accuracy(arr):
    accuracy = arr.sum() / 5.0 / 50.0
    return accuracy


def mean_reciprocal_rank(arr):
    temp = 0
    for row in arr:
        index = np.where(row == 1)
        if len(index[0]) > 0:
            temp += 1.0 / index_to_rank(index[0][0])

    return temp / 50.0

path = './IR_results/*'

file_list = glob.glob(path)

results = []

for f_name in file_list:

    df = pd.read_csv(f_name)
    result = df['label'].to_list()
    results.append(result)

results = np.array(results)
print('Top 5 accuracy')
print(top_5_accuracy(results))
print('MRR score')
print(mean_reciprocal_rank(results))


