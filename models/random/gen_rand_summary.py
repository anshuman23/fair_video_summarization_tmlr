import numpy as np
import pandas as pd
import json
import os
import h5py
import pickle

def get_vid_frames(vid_name, h5_dataset):
    f = h5py.File(h5_dataset, 'r')
    nframes = f[vid_name]['n_frames'][...]
    f.close()
    # frames = np.arange(nframes)
    return nframes

SPLITS_PATH = '../../splits/fvs_splits.json'
SUMMMARY_PATH = 'summaries'
DATA_PATH = '../../datasets/fvs.h5'
seed = 2732
np.random.seed(seed)

with open(SPLITS_PATH, 'r') as file:
    splits_data = json.load(file)

for split_idx,split in enumerate(splits_data):
    SPLIT_DIR = os.path.join(SUMMMARY_PATH, f'split{split_idx}')
    os.makedirs(SPLIT_DIR, exist_ok=True)
    # print(f' === SPLIT: {split_idx} ===')
    split_test_set = splits_data[split_idx]['test_keys']
    all_summaries = []
    # print(f'split{split_idx} : {split_test_set}')
    for vid_name in split_test_set:
        # print(f'== VIDEO : {vid_name} ==')
        nframes = get_vid_frames(vid_name, DATA_PATH)
        # 15% of video picked for summary
        k = int(0.15 * nframes)
        # randomly pick frames for summary
        picked_frames = np.random.choice(nframes, k, replace=False)
        # create summary
        summary = np.zeros(nframes)
        summary[picked_frames] = 1.0
        # print(len(summary))
        all_summaries.append(summary)
    # end of split
    # print('-'*20)
    with open(f'{SPLIT_DIR}/summaries.pkl', 'wb') as file:
        pickle.dump(all_summaries, file)
