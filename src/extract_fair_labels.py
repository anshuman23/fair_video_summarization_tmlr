import pandas as pd
import numpy as np
import json
import os

def sex(faces):
    if faces == 'none' or faces == ['None']:
        return None
    sexes = set(face.split('_')[1] for face in faces)
    if len(sexes) == 0:
        return None
    elif len(sexes) == 1:
        return sexes.pop()
    else:
        return list(sexes)

def ethnicity(faces):
    if faces == 'none' or faces == ['None']:
        return None
    # ethnicities = set(face.split('_')[0] for face in faces)
    ethnicities = set()
    for face in faces:
        eth = face.split('_')[0]
        if eth == 'SA' or eth == 'SE':
            ethnicities.add('AS')
        else:
            ethnicities.add(eth)
    if len(ethnicities) == 0:
        return None
    elif len(ethnicities) == 1:
        return ethnicities.pop()
    else:
        return list(ethnicities)
    

def num_unique_group(df, group_cat):
    _df = df.explode(group_cat)
    # get unique ind/sex/eth
    unique_groups = _df[group_cat].dropna().unique()
    num_unique = len(unique_groups)
    # assign int vals
    unique_dict = {value: index for index, value in enumerate(unique_groups)}
    return num_unique, unique_dict


def generate_npy(df, group_cat, num_unique, unique_dict):
    np_arr = np.zeros((len(df), num_unique))
    _df = df.explode(group_cat)
    _df = _df[['Frame', group_cat]].dropna(subset=[group_cat])
    for _, row in _df.iterrows():
        frame_idx = row['Frame']
        group_idx = unique_dict[row[group_cat]]
        np_arr[frame_idx, group_idx] = 1
    return np_arr


def save_npy(vid_num, group_cat, SAVE_DIR):
    SAVE_PATH = os.path.join(SAVE_DIR, group_cat)
    os.makedirs(SAVE_PATH, exist_ok=True)
    NP_SAVE_PATH = os.path.join(SAVE_PATH, f'video_{vid_num}.npy')
    np.save(NP_SAVE_PATH, np_arr)

FACES_PATH = 'faces_fvs_tvsum.csv'
SAVE_DIR = '../fair_npy_data'

faces_df = pd.read_csv(FACES_PATH)
faces_df.loc[:, 'Face'] = faces_df['Face'].apply(lambda x: eval(x) if isinstance(x, str) else x)

group_categories = ['sex', 'eth', 'ind'] # sex, ethnicity, individuals
video_ids = range(1,35) # 1 to 34 fvs video nums

for vid in video_ids:
    print(f' ==== VIDEO {vid} ====')
    # get fairness data for that video
    df = faces_df.loc[faces_df['Video'] == vid].copy()
    # Get Sex & Ethnicity IDS
    df['sex'] = df['Face'].apply(sex) 
    df['eth'] = df['Face'].apply(ethnicity)
    df.rename(columns={'Face': 'ind'}, inplace=True)
    df['ind'] = df['ind'].apply(lambda x: x if x != ['None'] else None)
    # print(df)
    # Get fairness npy for each group category
    for group_cat in group_categories:
        num_unique, unique_dict = num_unique_group(df, group_cat)
        print(f'{group_cat} : {num_unique} => {unique_dict}')
        # get array rep of fairness
        np_arr = generate_npy(df, group_cat, num_unique, unique_dict)
        print(f'{group_cat} shape: {np_arr.shape}')
        # save numpys
        save_npy(vid, group_cat, SAVE_DIR)
    print()
    