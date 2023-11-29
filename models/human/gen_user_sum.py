import numpy as np
import pandas as pd
import json
import os
import h5py
import pickle

def knapSack(W, wt, val, n):
	""" Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
	no concept of putting some part of item in the knapsack.

	:param int W: Maximum capacity -in frames- of the knapsack.
	:param list[int] wt: The weights (lengths -in frames-) of each video shot.
	:param list[float] val: The values (importance scores) of each video shot.
	:param int n: The number of the shots.
	:return: A list containing the indices of the selected shots.
	"""
	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i - 1] <= w:
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
			else:
				K[i][w] = K[i - 1][w]

	selected = []
	w = W
	for i in range(n, 0, -1):
		if K[i][w] != K[i - 1][w]:
			selected.insert(0, i - 1)
			w -= wt[i - 1]

	return selected

def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]    # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

        # Select the best shots using the knapsack implementation
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries

def get_user_summ(vid_name, h5_dataset):
    f = h5py.File(h5_dataset, 'r')
    user_summaries = f[vid_name]['user_summary'][...]
    f.close()
    # frames = np.arange(nframes)
    return user_summaries

def get_data(video_name, h5_dataset):
    with h5py.File(h5_dataset, 'r') as hdf:
        user_summary = np.array(hdf.get(video_name + '/user_summary'))
        cps = np.array(hdf.get(video_name + '/change_points'))
        n_frames = np.array(hdf.get(video_name + '/n_frames'))
        positions = np.array(hdf.get(video_name + '/picks'))
        gtscore = np.array(hdf.get(video_name + '/gtscore'))
    return user_summary, cps, n_frames, positions, gtscore

SPLITS_PATH = '../../splits/fvs_splits.json'
SUMMMARY_PATH = 'summaries'
DATA_PATH = '../../datasets/fvs.h5'

with open(SPLITS_PATH, 'r') as file:
    splits_data = json.load(file)

for split_idx,split in enumerate(splits_data):
    SPLIT_DIR = os.path.join(SUMMMARY_PATH, f'split{split_idx}')
    os.makedirs(SPLIT_DIR, exist_ok=True)
    # print(f' === SPLIT: {split_idx} ===')
    split_test_set = splits_data[split_idx]['test_keys']

    # print(f'split{split_idx} : {split_test_set}')
    all_gtscore, all_shot_bound, all_nframes, all_positions = [], [], [], []
    for vid_name in split_test_set:
        # take first user summary
        # user_summaries = get_user_summ(vid_name, DATA_PATH)[0]
        # all_summaries.append(user_summaries)
        # using gtscores
        user_summary, cps, n_frames, positions, gtscore = get_data(vid_name, DATA_PATH)
        all_shot_bound.append(cps)
        all_nframes.append(n_frames)
        all_positions.append(positions)
        all_gtscore.append(gtscore)

    all_summaries = generate_summary(all_shot_bound, all_gtscore, all_nframes, all_positions)

    with open(f'{SPLIT_DIR}/summaries.pkl', 'wb') as file:
        pickle.dump(all_summaries, file)
