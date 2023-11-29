from scipy.optimize import linprog
import numpy as np
import json
import os
import pickle

def select_samples(group_memberships, k):
    n_samples, n_groups = group_memberships.shape
    overall_group_proportions = np.sum(group_memberships, axis=0) / n_samples
    
    A_eq = group_memberships.T
    b_eq = overall_group_proportions * k
    
    bounds = [(0, 1)] * n_samples
    
    A_eq_k = np.ones((1, n_samples))
    b_eq_k = np.array([k])
    
    A_eq_combined = np.vstack((A_eq, A_eq_k))
    b_eq_combined = np.concatenate((b_eq, b_eq_k))
    
    res = linprog(c=np.zeros(n_samples), A_eq=A_eq_combined, b_eq=b_eq_combined, bounds=bounds)
    
    selected_indices = np.where(np.round(res.x) == 1)[0]
    
    selected_samples = group_memberships[selected_indices]
    selected_group_proportions = np.sum(selected_samples, axis=0) / len(selected_samples)
    
    return selected_samples, selected_indices, selected_group_proportions


def get_summary(group_memberships, k):
    selected_samples, selected_indices, selected_group_proportions = select_samples(group_memberships, k)
    overall_group_proportions = np.sum(group_memberships, axis=0) / len(group_memberships)
    print("Group Proportions (Selected Samples <Summary>):", selected_group_proportions)
    print("Group Proportions (Overall Dataset <GT proportions>):", overall_group_proportions)
    # print("Indices of Selected Samples:", selected_indices)
    print("Number of Samples Selected:", len(selected_indices))
    return selected_indices


# splits json path
SPLITS_PATH = '../../splits/fvs_splits.json'
SUMMMARY_PATH = 'summaries' # path to save summaries
DATA_PATH = '../../fair_npy_data'

group_categories = ['sex', 'eth', 'ind'] # sex, ethnicity, individuals

with open(SPLITS_PATH, 'r') as file:
    splits_data = json.load(file)

# for all 3 group scenarios
for group in group_categories:
    print('= * '*15)
    GROUP_PATH = os.path.join(SUMMMARY_PATH, group)
    GROUP_DATA_PATH = os.path.join(DATA_PATH, group)
    # get summaries for splits
    for split_idx,split in enumerate(splits_data):
        SPLIT_DIR = os.path.join(GROUP_PATH, f'split{split_idx}')
        os.makedirs(SPLIT_DIR, exist_ok=True)
        print(f' === SPLIT: {split_idx} | GROUP: {group} ===')
        split_test_set = splits_data[split_idx]['test_keys']
        all_summaries = []
        # print(f'split{split_idx} : {split_test_set}')
        for vid_name in split_test_set:
            print(f'== VIDEO : {vid_name} ==')
            vid_idx = int(vid_name.split('_')[1])
            VID_FAIR_DATA_PTH = os.path.join(GROUP_DATA_PATH, f'{vid_name}.npy')
            # get numpy fair data for vid
            vid_fair_data = np.load(VID_FAIR_DATA_PTH)
            vid_len, _ = vid_fair_data.shape
            # 15% vid summary
            k = int(0.15 * vid_len)
            print(f'vid len: {vid_len}, k: {k}')
            # LP-FVS
            selected_indices = get_summary(vid_fair_data, k)
            # convert to summary format
            summary = np.zeros(vid_len)
            summary[selected_indices] = 1.0
            # print(len(summary))
            all_summaries.append(summary)
        # end of split
        print('-'*20)
        with open(f'{SPLIT_DIR}/summaries.pkl', 'wb') as file:
            pickle.dump(all_summaries, file)

print('DONE')


