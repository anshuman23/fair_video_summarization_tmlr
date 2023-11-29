import h5py
import numpy as np
import json
import math
import os
import random
from tqdm import tqdm, trange
from pathlib import Path
import pprint
from scipy.stats import spearmanr, kendalltau, rankdata
from collections import Counter
from os import listdir
import argparse
import time
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------------
# DATALOADER
# --------------------------------
class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['../../datasets/summe.h5',
                         '../../datasets/tvsum.h5',
                         '../../datasets/fvs.h5']
        # self.filename = '../H5_file/fvs.h5'
        # self.splits_filename = ['../splits/splits.json'] #'data/splits/' + self.name + '_splits.json']
        self.splits_filename = ['../../splits/' + self.name + '_splits.json']
        self.split_index = split_index

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        elif 'fvs' in self.splits_filename[0]:
            self.filename = self.datasets[2]

        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_frame_features.append(frame_features)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len
    
    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features
        test  mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        frame_features = self.list_frame_features[index]

        if self.mode == 'test':
            video_name = self.split[self.mode + '_keys'][index]
            return frame_features, video_name
        else:
            return frame_features


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


# --------------------------------
# Model Setup
# --------------------------------
class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """ The basic Attention 'cell' containing the learnable parameters of Q, K and V.

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.Wk = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wv = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size+2, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_entropy(logits):
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T].
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[0])

    def forward(self, x):
        """ Compute the weighted frame features, through the Block diagonal sparse attention matrix and the estimates of
        the frames attentive uniqueness and the diversity.

        :param torch.Tensor x: Frame features with shape [T, input_size].
        :return: A tuple of:
                    y: The computed weighted features, with shape [T, input_size].
                    att_win : The Block diagonal sparse attention matrix, with shape [T, T].
        """
        # Compute the pairwise dissimilarity of each frame, on the initial feature space (GoogleNet features)
        x_unit = F.normalize(x, p=2, dim=1)
        similarity = x_unit @ x_unit.t()
        diversity = 1 - similarity

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        energies = torch.matmul(Q, K.transpose(1, 0))
        att_weights = self.softmax(energies)

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=energies)
        entropy = F.normalize(entropy, p=1, dim=-1)

        # Compute the mask to form the Block diagonal sparse attention matrix
        D = self.block_size
        num_blocks = math.ceil(energies.shape[0] / D)
        keepingMask = torch.ones(num_blocks, D, D, device=att_weights.device)
        keepingMask = torch.block_diag(*keepingMask)[:att_weights.shape[0], :att_weights.shape[0]]
        zeroingMask = (1 - keepingMask)
        att_win = att_weights * keepingMask

        # Pick those frames that are "invisible" to a frame, aka outside the block (mask)
        attn_remainder = att_weights * zeroingMask
        div_remainder = diversity * zeroingMask

        # Compute non-local dependencies based on the diversity of those frames
        dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
        dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
        masked_dep_factor = dep_factor * keepingMask
        att_win += masked_dep_factor

        y = torch.matmul(att_win, V)
        characteristics = (entropy, dep_factor[0, :])
        characteristics = torch.stack(characteristics).detach()
        outputs = torch.cat(tensors=(y, characteristics.t()), dim=-1)

        y = self.out(outputs)
        return y, att_win.clone()



class CA_SUM(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """ Class wrapping the CA-SUM model; its key modules and parameters.
        
        :param int input_size: The expected input feature size.
        :param int output_size: The produced output feature size.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(CA_SUM, self).__init__()

        self.attention = SelfAttention(input_size=input_size, output_size=output_size, block_size=block_size)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features):
        """ Produce frame-level importance scores from the frame features, using the CA-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by 
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        residual = frame_features
        weighted_value, attn_weights = self.attention(frame_features)
        y = residual + weighted_value
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights


# --------------------------------
# Train Solver
# --------------------------------
class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """ Class that Builds, Trains and Evaluates CA-SUM model. """
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the CA-SUM model, its key modules and parameters. """
        # Model creation
        self.model = CA_SUM(input_size=self.config.input_size,
                            output_size=self.config.input_size,
                            block_size=self.config.block_size).to(self.config.device)

        if self.config.init_type is not None:
            self.init_weights(net=self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            #! Tensorboard
            #! self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def length_regularization_loss(self, scores):
        """ Compute the summary-length regularization loss based on eq. (1).

        :param torch.Tensor scores: Frame-level importance scores, produced by our CA-SUM model.
        :return: A (torch.Tensor) value indicating the summary-length regularization loss.
        """
        return torch.abs(torch.mean(scores) - self.config.reg_factor)

    def train(self):
        """ Main function to train the CA-SUM model. """
        if self.config.verbose:
            tqdm.write('Time to train the model...')

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
                self.optimizer.zero_grad()

                for _ in trange(self.config.batch_size, desc='Video', ncols=80, leave=False):
                    frame_features = next(iterator)
                    frame_features = frame_features.squeeze(0).to(self.config.device)
                    output, _ = self.model(frame_features)
                    loss = self.length_regularization_loss(output)
                    loss_history.append(loss.data)
                    loss.backward()

                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()
            if self.config.verbose:
                tqdm.write(f'[{epoch_i}] loss: {loss.item()}')

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            #! Tensorboard     
            #! self.writer.update_loss(loss, epoch_i, 'loss_epoch')

            # Uncomment to save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            if self.config.verbose:
                tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i, save_weights=False):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        :param bool save_weights: Optionally, the user can choose to save the attention weights in a (large) h5 file.
        """
        self.model.eval()

        weights_save_path = self.config.score_dir.joinpath("weights.h5")
        out_scores_dict = {}
        for frame_features, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)

            with torch.no_grad():
                scores, attn_weights = self.model(frame_features)  # [1, seq_len]
                scores = scores.squeeze(0).cpu().numpy().tolist()
                attn_weights = attn_weights.cpu().numpy()

                out_scores_dict[video_name] = scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            scores_save_path = self.config.score_dir.joinpath(f"{self.config.video_type}_{epoch_i}.json")
            with open(scores_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(scores_save_path)}.')
                json.dump(out_scores_dict, f)
            scores_save_path.chmod(0o777)

            if save_weights and (epoch_i+1 == self.config.n_epochs or epoch_i+1 == 0):
                with h5py.File(weights_save_path, 'a') as weights:
                    weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)


# --------------------------------
# Train Configs
# --------------------------------
exp_dir_name = 'exp2'
save_dir = Path(exp_dir_name)


def str2bool(v):
    """ Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """ Configuration Class: set kwargs as class attributes with setattr. """
        self.log_dir, self.score_dir, self.save_dir = None, None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.reg_factor, self.video_type)

    def set_dataset_dir(self, reg_factor=0.6, video_type='wholeh5'):
        """ Function that sets as class attributes the necessary directories for logging important training information.

        :param float reg_factor: The utilized length regularization factor.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        """
        self.log_dir = save_dir.joinpath('reg' + str(reg_factor), video_type, 'logs/split' + str(self.split_index))
        self.score_dir = save_dir.joinpath('reg' + str(reg_factor), video_type, 'results/split' + str(self.split_index))
        self.save_dir = save_dir.joinpath('reg' + str(reg_factor), video_type, 'models/split' + str(self.split_index))

    def __repr__(self):
        """ Pretty-print configurations in alphabetical order. """
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


# def get_config(parse=True, **optional_kwargs):
#     """ Get configurations as attributes of class
#         1. Parse configurations with argparse.
#         2. Create Config class initialized with parsed kwargs.
#         3. Return Config class.
#     """
#     kwargs = {
#         'mode' : "train",
#         'verbose' : False,
#         'video_type' : "fvs",  # change to allinone for original dataset
#         'input_size' : 1024,
#         'block_size' : 10,   ####
#         'init_type' : "xavier",
#         'init_gain' : 1.0,   # reduce the init_gain to 1.0
#         'n_epochs' : 200,
#         'batch_size' : 4, 
#         'seed' : 12345,
#         'clip' : 1.0,        ####
#         'lr' : 1e-4,         
#         'l2_req' : 1e-6,     
#         'reg_factor' : 5.0,  ####
#         'split_index' : _SPLIT #0
#     }

#     kwargs.update(optional_kwargs)

#     return Config(**kwargs)

def get_config(parse=True, **optional_kwargs):
    """ Get configurations as attributes of class
        1. Parse configurations with argparse.
        2. Create Config class initialized with parsed kwargs.
        3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train', help='Mode for the configuration [train | test]')
    parser.add_argument('--verbose', type=str2bool, default='false', help='Print or not training messages')
    parser.add_argument('--video_type', type=str, default='fvs', help='Dataset to be used')

    # Model
    parser.add_argument('--input_size', type=int, default=1024, help='Feature size expected in the input')
    parser.add_argument('--block_size', type=int, default=10, help="Size of blocks used inside the attention matrix") # default=60
    parser.add_argument('--init_type', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=1.0, help='Scaling factor for the initialization methods') # 1.4142

    # Train
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs') # 400
    parser.add_argument('--batch_size', type=int, default=20, help='Size of each batch in training')
    parser.add_argument('--seed', type=int, default=12345, help='Chosen seed for generating random numbers')
    parser.add_argument('--clip', type=float, default=1.0, help='Max norm of the gradients') # 5.0
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate used for the modules') # 5e-4
    parser.add_argument('--l2_req', type=float, default=1e-6, help='Weight regularization factor') # 1e-5
    parser.add_argument('--reg_factor', type=float, default=5.0, help='Length regularization factor') # 0.6
    parser.add_argument('--split_index', type=int, default=0, help='Data split to be used [0-4]')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)

# 2 fps
config = get_config(mode='train')
test_config = get_config(mode='test')

print('Currently selected split_index:', config.split_index)
# train_loader = get_loader('train', "fvs", _SPLIT) 
# test_loader = get_loader('test', "fvs", _SPLIT)
train_loader = get_loader(config.mode, config.video_type, config.split_index)
test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
solver = Solver(config, train_loader, test_loader)

print('Training...')
TRAIN_TIME = time.time()

solver.build()
solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
solver.train()

elapsed_time = time.time() - TRAIN_TIME
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Train time: {hours} hrs, {minutes} mins, {seconds} s")


# --------------------------------
# Generate all json files
# --------------------------------
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
    
# print('Evaluation:')
# path = 'exp1/reg5.0/fvs/results/split0'  ######### change to allinone.h5 for og dataset
# path = 'exp2/reg5.0/fvs/results/split' + str(_SPLIT)
path = f'{save_dir}/reg5.0/{config.video_type}/results/split{config.split_index}'
# dataset = "fvs" #'googlenet_2ps' ######### change to allinone.h5 for og dataset
dataset = config.video_type
eval_method = 'avg'

results = [f for f in listdir(path) if f.endswith(".json")]
# results.sort(key=lambda video: int(video[6:-5]))
results.sort(key=lambda ep: int(ep.split('_')[-1].split('.')[0]))
dataset_path = f'../../datasets/{config.video_type}.h5' ######## change to allinone for og dataset

# summary_dir = 'exp2/reg5.0/fvs/summaries/split' + str(_SPLIT)
SUMMARY_DIR = save_dir.joinpath(config.video_type, 'summaries/split' + str(config.split_index))
os.makedirs(SUMMARY_DIR, exist_ok=True)

f_score_epochs = []
summaries_epoch = []
# for epoch in results:                       # for each epoch ...
for epoch in tqdm(results):
    all_scores = []
    with open(path + '/' + epoch) as f:     # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:                    # for each video inside that json file ...
            scores = np.asarray(data[video_name])  # read the importance scores from frames
            all_scores.append(scores)

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
    with h5py.File(dataset_path, 'r') as hdf:
        for video_name in keys:
            user_summary = np.array(hdf.get(video_name + '/user_summary'))
            sb = np.array(hdf.get(video_name + '/change_points'))
            n_frames = np.array(hdf.get(video_name + '/n_frames'))
            positions = np.array(hdf.get(video_name + '/picks'))

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, eval_method)
        all_f_scores.append(f_score)


    f_score_epochs.append(np.mean(all_f_scores))
    # num_epoch = epoch.split(".")[0][6:]
    # num_epoch = epoch.split("_")[1].split(".")[0]
    # print(f"[epoch {num_epoch}] f_score: {np.mean(all_f_scores)}")
    # np.save(f'summaries/summaries_{num_epoch}.npy', all_summaries)
    # with open(f'{summary_dir}/summaries_{num_epoch}.pkl', 'wb') as file:
    #     pickle.dump(all_summaries, file)
    summaries_epoch.append(all_summaries)

f = max(f_score_epochs)   # returns the best f score
i = f_score_epochs.index(f)  # returns the best epoch number
best_summaries = summaries_epoch[i] # get summaries output for the best epoch
# save best summary
with open(f'{SUMMARY_DIR}/best_summaries.pkl', 'wb') as file:
    pickle.dump(best_summaries, file)

# print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Summary: {summary_dir}/summaries_{i}.pkl")
print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Best Summary: {SUMMARY_DIR}/best_summaries.pkl")
# Save the importance scores in txt format.
with open(path + '/f_scores.txt', 'w') as outfile:
    for f_score in f_score_epochs:
        outfile.write('%s\n' % f_score)

