"""
Author: Xiangyi
"""

import pandas as pd
from stackoverflow_data_loader import SOFDataLoader

from caption_data_loader import CaptionDataLoader
data_loader = CaptionDataLoader()
data_loader.load()


def sec_to_string(sec: int):
    sec_str = str(sec)
    sec_str = (5 - len(sec_str)) * '0' + sec_str
    return sec_str


# Find all related IMAGE filename based on action timestamp.
img_paths = []
for video_num, v_dict in data_loader.action_caption_dict.items():
    for video_sec in v_dict.keys():
        img_path = video_num + '/' + sec_to_string(video_sec) + '.jpg'
        img_paths.append(img_path)


divide_at = int(len(img_paths) / 10 * 8)

test_img_paths = img_paths[divide_at:]

f_real = open('real_caption.txt', 'r+')
f_pred = open('pred_caption.txt', 'r+')

tuple_list = []
for p in test_img_paths:
    real = f_real.readline()
    pred = f_pred.readline()
    tuple_list.append((p, real[8:-8], pred[:-2]))

f_real.close()
f_pred.close()

df = pd.DataFrame(tuple_list, columns=['Image_Name', 'Real_Caption', 'Pred_Caption']).astype(str)
df.to_csv('filename_real_pred.csv')


# select 200 caption for user study
select_df = df[df['Pred_Caption'].str.len() > 100]
print(select_df.shape)
select_df = select_df.sample(n=50, random_state=826).sort_index()
select_df.to_csv('selected_caption.csv')
# print(select_df.to_string())

# stackoverflow_ir_system = SOFDataLoader()
print('IR system done.')

filename_to_ir_result = {}
pre_path = 'IR_results/'
for row in select_df.itertuples(index=True, name='Pandas'):
    print('////')
    filename = pre_path + getattr(row, 'Image_Name')[:-3] + 'csv'
    filename = filename.replace('/', '_')

    print(filename)
    pred_caption = getattr(row, 'Pred_Caption')
    print(pred_caption)
    #top_5_results = stackoverflow_ir_system.compute_bm25(pred_caption)
    #top_5_results.to_csv(filename, index=False)





