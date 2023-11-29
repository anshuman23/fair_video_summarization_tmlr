import numpy as np
import json
import os
import pickle
import h5py
import argparse
import string

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

def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).
    :param np.ndarray predicted_summary: The generated summary from our model.
    :param np.ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)


def get_pred_summary(PATH):
    files = os.listdir(PATH)
    if not files:
        return False
    pkl_file = next(file for file in files if file.endswith('.pkl'))
    file_path = os.path.join(PATH, pkl_file)
    with open(file_path, 'rb') as file:
        all_summaries = pickle.load(file)
    return all_summaries

def get_user_summary(split_test_set, PATH):
    all_user_summaries = []
    # read h5 dataset
    with h5py.File(PATH, 'r') as hdf: 
        # get user summareis for all vids in split       
        for vid_name in split_test_set:
            user_summary = np.array( hdf.get(f'{vid_name}/user_summary') )
            all_user_summaries.append(user_summary)
    return all_user_summaries

def fscore(all_summaries, all_user_summaries):
    # for sum in all_user_summaries:
    #     print(sum.shape)
    assert len(all_summaries) == len(all_user_summaries)
    all_f_scores = []
    # Get avg f score for this split/fold
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summaries[video_index]
        f_score = evaluate_summary(summary, user_summary, 'avg')  
        all_f_scores.append(f_score)
    # Get mean f score
    mean_f_score = np.mean(all_f_scores)
    return mean_f_score, all_f_scores

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

def get_split_fscore(model_name, SUMMARIES_PATH,SPLITS_PATH, FVS_DATASET_PATH):
    all_splits_results = []
    with open(SPLITS_PATH, 'r') as file:
        splits_data = json.load(file)
    for split_idx, split in enumerate(splits_data):
        split_results = []
        split_results.append(model_name)
        split_results.append(split_idx)
        # get dir for split
        SPLIT_DIR = os.path.join(SUMMARIES_PATH, f'split{split_idx}')
        # get split test set
        split_test_set = splits_data[split_idx]['test_keys']
        # Get GT user summary
        all_user_summaries = get_user_summary(split_test_set, FVS_DATASET_PATH)
        # Get predicted summary
        all_summaries = get_pred_summary(SPLIT_DIR)
        if not all_summaries:
            continue
        # get mean fscore for split
        mean_fscore, all_f_scores = fscore(all_summaries, all_user_summaries)
        # split_f_scores = ["{:.3f}".format(f_score) for f_score in all_f_scores]
        # print('F-SCORES: ',split_f_scores)
        # print(f'MEAN F-SCORE: {mean_fscore:.3f}')
        split_results.append(mean_fscore)
        all_splits_results.append(split_results)
    return all_splits_results


def get_all_fscore(model_name, SUMMARIES_PATH,SPLITS_PATH, FVS_DATASET_PATH):
    # all splits fscores for model
    all_splits_results = get_split_fscore(model_name, SUMMARIES_PATH,SPLITS_PATH, FVS_DATASET_PATH)
    # get average fscores for each split
    fscores = [split[2] for split in all_splits_results]
    # mean fscore across all splits
    mean_fscore = np.mean(fscores)
    # print table for each model
    all_splits_results = [["Model", "Split", "F-score"]] + all_splits_results
    print_table(all_splits_results)
    print()
    # return mean fscore for model
    return mean_fscore



# exit()
parser = argparse.ArgumentParser()
model_options = ['all', 'random', 'human', 'ca-sum', 'ac-sum-gan', 'sum-gan-aae', 'sum-gan-sl', 'sum-ind',
                 'dsnet', 'vasnet', 'pgl-sum', 'lp-fvs_sex', 'lp-fvs_eth', 'lp-fvs_ind']
parser.add_argument('-m', '--model', type=str, choices=model_options, required=True, help="model name")
args = parser.parse_args()

SPLITS_PATH = '../splits/fvs_splits.json'
FVS_DATASET_PATH = '../datasets/fvs.h5'

model_fscores = []
if args.model == 'all':
    for model in model_options:
        if model == 'all':
            continue
        model_fscore = []
        model_fscore.append(model)
        SUMMARIES_PATH = get_summ_path(model)
        mean_fscore = get_all_fscore(model, SUMMARIES_PATH, SPLITS_PATH, FVS_DATASET_PATH)
        model_fscore.append(mean_fscore)
        model_fscores.append(model_fscore)
else:
    model_fscore = []
    model_fscore.append(args.model)
    SUMMARIES_PATH = get_summ_path(args.model)
    mean_fscore = get_all_fscore(args.model, SUMMARIES_PATH, SPLITS_PATH, FVS_DATASET_PATH)
    model_fscore.append(mean_fscore)
    model_fscores.append(model_fscore)

print(' === Mean results for all splits ===', end='\n\n')
model_fscores = [["Model", "F-score"]] + model_fscores
# print_table(model_fscores, cell_width=[10,20,2])
print_table(model_fscores)

