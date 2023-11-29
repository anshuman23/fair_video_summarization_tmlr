import numpy as np
import pandas as pd
import json
import pickle
import os
import argparse
import string

VERBOSE = False

def print_table(data):
    if not data:  # Check if the data is not empty
        return
    # get the max length of each column (considering also the headers)
    column_widths = [max(len(str(x)) for x in col) for col in zip(*data)]
    # create a format string that will space out the columns correctly
    print(''.join('-' * (width + 3) for width in column_widths))
    format_string = '   '.join(['{:<' + str(width) + '}' for width in column_widths])
    # print each row using the format string
    for i, row in enumerate(data):
        print(format_string.format(*row))
        if i == 0:  # after printing the first row, print the dashed line
            print(''.join('-' * (width + 3) for width in column_widths))

    # print the dashed line at the end of the table
    print(''.join('-' * (width + 3) for width in column_widths))


def SumBal(overall_group_props, summary_group_props):
    assert len(overall_group_props) == len(summary_group_props)
    # to track violating group
    sumBal_list = []
    # go through all groups and get their proportions
    for group_id, overall_group_prop in enumerate(overall_group_props):
        # get R for a group
        # R_g = overall_group_prop / summary_group_props[group_id]
        if summary_group_props[group_id] != 0:
            R_g = overall_group_prop / summary_group_props[group_id]
            # get sumBal for group/individual
            sumBal_g = min(R_g, 1/R_g)
        else:
            sumBal_g = 0
        if VERBOSE:
            print(f'SumBal({group_id})={sumBal_g:.3f} ', end='|')
        sumBal_list.append(sumBal_g)
    # Get SumBal
    sumBal = min(sumBal_list)
    # get violating group
    group_id = sumBal_list.index(sumBal)
    if VERBOSE:
        print()
        print(f'\tSumBal: {sumBal:.3f}  , violating: {group_id}')
    # return sumbak score and violating group/individual
    return sumBal, group_id


def get_group_proportion(H):
    n_samples,groups = H.shape
    group_props = []
    for i in range(groups):
        group = H[:,i]
        prop = np.sum(group) / n_samples
        group_props.append(prop)
        if VERBOSE:
            print(f'{i}: {prop:.3f} ', end='|')
    if VERBOSE:
        print()
    return group_props

def get_user_sum(split_idx, vid_idx, SUMMARIES_PATH):
    # get approprate pkl file for split 
    PATH = os.path.join(SUMMARIES_PATH, f'split{split_idx}')
    files = os.listdir(PATH)
    try:
        pkl_file = next(file for file in files if file.endswith('.pkl'))
    except:
        return None
    file_path = os.path.join(PATH, pkl_file)
    with open(file_path, 'rb') as file:
        all_summaries = pickle.load(file)
    # get appropriate video summary within all summaries
    full_summary = all_summaries[vid_idx] 
    # get only select indeces
    select_indices = np.where(full_summary == 1)[0]
    # print(summary.shape)
    return select_indices

def get_summ_path(model):
    # get SUMMARIES PATH
    if 'fvs' in model:
        # print(model)
        parts = model.split('_')
        model_name = parts[0]
        group = parts[1]
        SUMMARIES_PATH = f'../models/{model_name}/summaries/{group}'
    elif model == 'dsnet':
        SUMMARIES_PATH = f'../models/{model}/src/models/fvs/summaries'
    elif model == 'vasnet':
        SUMMARIES_PATH = f'../models/{model}/VASNet/data/summaries'
    elif model == 'random' or model == 'human' or model == 'sum-ind':
        SUMMARIES_PATH = f'../models/{model}/summaries'
    else:
        SUMMARIES_PATH = f'../models/{model}/exp2/fvs/summaries'
    return SUMMARIES_PATH


def video_sumbal(vid_name, vid_idx, split_idx, GROUP_DATA_PATH):
    if VERBOSE:
        print('VIDEO: ', vid_name)
    VID_FAIR_DATA_PTH = os.path.join(GROUP_DATA_PATH, f'{vid_name}.npy')
    # vid fair npy data for group
    overall_gender_memberships = np.load(VID_FAIR_DATA_PTH)
    if VERBOSE:
        print('\tOverall Props --> ', end='')
    overall_group_props = get_group_proportion(overall_gender_memberships)
    # pred summary (selected indexes from summary)
    pred_summary_idxs = get_user_sum(split_idx, vid_idx, SUMMARIES_PATH)
    # Get M/F - group membership information for the selected indexes
    summary_gender_memberships = overall_gender_memberships[pred_summary_idxs]
    # get group proportions for summary
    if VERBOSE:
        print('\tSummary Props --> ', end='|')
    summary_group_props = get_group_proportion(summary_gender_memberships)
    # Print lengths (n and k)
    n = overall_gender_memberships.shape[0]
    k = summary_gender_memberships.shape[0]
    if VERBOSE:
        print(f'\tn: {n} , k: {k} , %: {(k/n):.2f}')
        print('\tSumBal(Group/Ind) --> ', end='|')
    sumBal, violating_id = SumBal(overall_group_props, summary_group_props)
    # split_all_sumbals.append(sumBal)
    return (sumBal, violating_id, vid_name)


def split_sumbal(splits_data, group, GROUP_DATA_PATH):
    split_data = []
    for split_idx,_ in enumerate(splits_data):
        if VERBOSE:
            print(f'========= SPLIT: {split_idx} | GROUP: {group} =========')
        # SPLIT_DIR = os.path.join(SUMMARIES_PATH, f'split{split_idx}')
        split_test_set = splits_data[split_idx]['test_keys']
        if VERBOSE:
            print('  VIDS: ',split_test_set, end='\n\n')
        splits_sumbals = []
        for vid_idx, vid_name in enumerate(split_test_set):
            sumbal_data = video_sumbal(vid_name, vid_idx, split_idx, GROUP_DATA_PATH)
            splits_sumbals.append(sumbal_data)
        # split complete
        min_sumbal = min(splits_sumbals, key=lambda x: x[0])
        avg_sumbal = sum(t[0] for t in splits_sumbals) / len(splits_sumbals)
        # print(splits_sumbals)
        # print(min_sumbal)
        # print(avg_sumbal)
        split_data.append((avg_sumbal, min_sumbal))
    return split_data


def full_data(model, group, split_data):
    # data row
    results = []
    # Model, Group, Split idx, avg sumbal, Min sumbal, violating
    
    # append per splits data
    for split_idx, split in enumerate(split_data):
        split_results = []
        split_results.append(model)
        split_results.append(group)
        split_results.append(split_idx)
        # avg = split[0]
        split_results.append(split[0])
        min_sb = split[1]
        # min value
        split_results.append(min_sb[0])
        # violating group
        viol = f'{min_sb[2]}:{min_sb[1]}'
        split_results.append(viol)
        results.append(split_results)

    # average
    avg_results = []
    avg_results.append(model)
    avg_results.append(group)
    # average for all spits
    avg_results.append('all')
    # average
    avg_split_sumbal = sum(t[0] for t in split_data) / len(split_data)
    avg_results.append(avg_split_sumbal)
    min_split_sumbal = min(split_data, key=lambda x: x[0])[1]
    # min val
    avg_results.append(min_split_sumbal[0])
    # viola
    split_viol = f'{min_split_sumbal[2]}:{min_split_sumbal[1]}'
    avg_results.append(split_viol)
    results.append(avg_results)

    return results



def group_sumbal(model, group_categories, SUMMARIES_PATH):
    # open splits json to get test set info
    with open(SPLITS_PATH, 'r') as file:
        splits_data = json.load(file)

    all_results = []
    # for all 3 group scenarios (sex,eth,ind)
    for group in group_categories:
        GROUP_DATA_PATH = os.path.join(DATA_PATH, group)
        # get splits sumbals
        split_data = split_sumbal(splits_data, group, GROUP_DATA_PATH)
        results = full_data(model, group, split_data)
        results = [["Model", "Group", "Split", "Avg SumBal", "Min SumBal", "Violating"]] + results
        print_table(results)
        print()
    # return all_results


parser = argparse.ArgumentParser()
model_options = ['all', 'random', 'human', 'ca-sum', 'ac-sum-gan', 'sum-gan-aae', 'sum-gan-sl', 'sum-ind',
                 'dsnet','vasnet', 'pgl-sum', 'lp-fvs_sex', 'lp-fvs_eth', 'lp-fvs_ind']
parser.add_argument('-m', '--model', type=str, choices=model_options, required=True, help="model name")
args = parser.parse_args()

SPLITS_PATH = '../splits/fvs_splits.json'
FVS_DATASET_PATH = '../datasets/fvs.h5'
DATA_PATH = '../fair_npy_data'

group_categories = ['sex', 'eth', 'ind'] # sex, ethnicity, individuals

model_sumbals = []
if args.model == 'all':
    for model in model_options:
        if model == 'all':
            continue
        model_sb = []
        model_sb.append(model)
        print('='*50)
        print('\t\t', model.upper())
        print('='*50)
        SUMMARIES_PATH = get_summ_path(model)
        group_sumbal(model, group_categories, SUMMARIES_PATH)

