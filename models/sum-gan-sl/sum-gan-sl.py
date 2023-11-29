
import h5py
import numpy as np
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import pprint
import time
import pickle
import numpy as np
from os import listdir
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# --------------------------------
# DATALOADER
# --------------------------------
class VideoData(Dataset):
    def __init__(self, mode, split_index):
        self.mode = mode
        self.name = 'fvs'
        # self.name = video_type.lower()
        # self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
        #                  '/content/tvsum_short.h5',
        #                  '/content/drive/MyDrive/fair_vid_sum/H5_file/allinone.h5'] ######### change to gn_full.h5 for og dataset googlenet_2ps.h5
        self.datasets = ['../../datasets/summe.h5',
                         '../../datasets/tvsum.h5',
                         '../../datasets/fvs.h5']
        # self.filename = '../H5_file/fvs.h5'
        # self.splits_filename = ['../splits/splits.json']
        self.splits_filename = ['../../splits/' + self.name + '_splits.json']
        self.splits = []
        self.split_index = split_index # it represents the current split (varies from 0 to 4)
        temp = {}

        # if 'summe' in self.splits_filename[0]:
        #     self.filename = self.datasets[0]
        # elif 'tvsum' in self.splits_filename[0]:
        #     self.filename = self.datasets[1]
        # else:
        #   self.filename = self.datasets[2]
        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        elif 'fvs' in self.splits_filename[0]:
            self.filename = self.datasets[2]

        self.video_data = h5py.File(self.filename, 'r')

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for split in data:
                temp['train_keys'] = split['train_keys']
                temp['test_keys'] = split['test_keys']
                self.splits.append(temp.copy())

    def __len__(self):
        self.len = len(self.splits[0][self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.splits[self.split_index][self.mode + '_keys'][index]
        frame_features = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features


def get_loader(mode, split_index):
    if mode.lower() == 'train':
        vd = VideoData(mode, split_index)
        return DataLoader(vd, batch_size=1)
    else:
        return VideoData(mode, split_index)


# --------------------------------
# MODEL
# --------------------------------

class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list
    
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 500] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            last hidden
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=500, hidden_size=500, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [num_layers=2, 1, hidden_size] (h from all layers)
            # c: [num_layers=2, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda()

        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h_mu: [num_layers=2, hidden_size]
            h_log_variance: [num_layers=2, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        """
        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        # [num_layers, hidden_size]
        h = h.squeeze(1)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features):
        """
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            h_mu: [num_layers=2, hidden_size]
            h_log_variance: [num_layers=2, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        """

        # Apply weights
        # [seq_len, 1]
        scores = self.s_lstm(image_features)

        # [seq_len, 1, hidden_size]
        weighted_features = image_features * scores.view(-1, 1, 1)

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features
    

class cLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, input_size]
        Return:
            last_h: [1, hidden_size]
        """
        self.lstm.flatten_parameters()

        # output: [seq_len, batch, hidden_size * num_directions]
        # h_n, c_n: [num_layers * num_directions, batch_size, hidden_size]
        output, (h_n, c_n) = self.lstm(features, init_hidden)

        # [batch_size, hidden_size]
        last_h = h_n[-1]

        return last_h


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator: cLSTM + output projection to probability"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h : [1, hidden_size]
                Last h from top layer of discriminator
            prob: [batch_size=1, 1]
                Probability to be original feature from CNN
        """

        # [1, hidden_size]
        h = self.cLSTM(features)

        # [1]
        prob = self.out(h).squeeze()

        return h, prob
    
# --------------------------------
# TRAIN CONFIG
# --------------------------------
# labels for training the GAN part of the model
original_label = torch.tensor(1.0).cuda()
summary_label = torch.tensor(0.0).cuda()

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN-sl model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters())
                + list(self.summarizer.vae.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.summarizer.vae.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)

            # self.writer = TensorboardWriter(str(self.config.log_dir))

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_sum, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    criterion = nn.MSELoss()

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_original_loss_history = []
            c_summary_loss_history = []
            for batch_i, image_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                self.model.train()

                # [batch_size=1, seq_len, 1024]
                # [seq_len, 1024]
                image_features = image_features.view(-1, self.config.input_size)

                # [seq_len, 1024]
                image_features_ = Variable(image_features).cuda()

                #---- Train sLSTM, eLSTM ----#
                if self.config.verbose:
                    tqdm.write('\nTraining sLSTM and eLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer(original_features)

                h_origin, original_prob = self.discriminator(original_features)
                h_sum, sum_prob = self.discriminator(generated_features)

                if self.config.verbose:
                    tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                prior_loss = self.prior_loss(h_mu, h_log_variance)
                sparsity_loss = self.sparsity_loss(scores)

                if self.config.verbose:
                    tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f}')

                s_e_loss = reconstruction_loss + prior_loss + sparsity_loss

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()
                # Gradient cliping
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip) # depreciated
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()

                s_e_loss_history.append(s_e_loss.data)

                #---- Train dLSTM (generator) ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer(original_features)

                h_origin, original_prob = self.discriminator(original_features)
                h_sum, sum_prob = self.discriminator(generated_features)
                
                if self.config.verbose:
                    tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                g_loss = self.criterion(sum_prob, original_label)

                if self.config.verbose:
                    tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, g loss: {g_loss.item():.3f}')

                d_loss = reconstruction_loss + g_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()
                # Gradient cliping
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip) # depreciated
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.d_optimizer.step()

                d_loss_history.append(d_loss.data)

                #---- Train cLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training cLSTM...')

                self.c_optimizer.zero_grad()

                # Train with original loss
                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                h_origin, original_prob = self.discriminator(original_features)
                c_original_loss = self.criterion(original_prob, original_label)
                c_original_loss.backward()

                # Train with summary loss
                scores, h_mu, h_log_variance, generated_features = self.summarizer(original_features)
                h_sum, sum_prob = self.discriminator(generated_features.detach())
                c_summary_loss = self.criterion(sum_prob, summary_label)
                c_summary_loss.backward()
                
                if self.config.verbose:
                    tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
                    tqdm.write(f'gen loss: {g_loss.item():.3f}')
                
                # Gradient cliping
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip) # depreciated
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.c_optimizer.step()

                c_original_loss_history.append(c_original_loss.data)
                c_summary_loss_history.append(c_summary_loss.data)
                

                if self.config.verbose:
                    tqdm.write('Plotting...')

                # self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                # self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                # self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                # self.writer.update_loss(g_loss.data, step, 'gen_loss')

                # self.writer.update_loss(original_prob.data, step, 'original_prob')
                # self.writer.update_loss(sum_prob.data, step, 'sum_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_original_loss = torch.stack(c_original_loss_history).mean()
            c_summary_loss = torch.stack(c_summary_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            # self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            # self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            # self.writer.update_loss(c_original_loss, step, 'c_original_loss')
            # self.writer.update_loss(c_summary_loss, step, 'c_summary_loss')

            # Save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            if self.config.verbose:
                tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)


    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for video_tensor, video_name in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, batch=1, 1024]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor).cuda()

            # [seq_len, 1, hidden_size]
            video_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)

            # [seq_len]
            with torch.no_grad():
                scores = self.summarizer.s_lstm(video_feature).squeeze(1)
                scores = scores.cpu().numpy().tolist()

                out_dict[video_name] = scores
                
            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            score_save_path = self.config.score_dir.joinpath(
                f'{self.config.video_type}_{epoch_i}.json')
            with open(score_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)


# --------------------------------
# Train Configs
# --------------------------------
exp_dir_name = 'exp2'
save_dir = Path(exp_dir_name)
# save_dir = Path('exp0')

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='SumMe'):
        self.log_dir = save_dir.joinpath(video_type, 'logs/split'+str(self.split_index))
        self.score_dir = save_dir.joinpath(video_type, 'results/split'+str(self.split_index))
        self.save_dir = save_dir.joinpath(video_type, 'models/split'+str(self.split_index))

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default='false')
    parser.add_argument('--video_type', type=str, default='fvs')

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512) # 500
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--regularization_factor', type=float, default=5.0) # 0.1

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--clip', type=float, default=1.0) # 5.0
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--split_index', type=int, default=0)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    # kwargs = {
    #     'mode' : mode,
    #     'verbose' : False,
    #     'video_type' : 'fvs',
    #     'input_size' : 1024,
    #     'hidden_size' : 500,
    #     'num_layers' : 2,
    #     'regularization_factor': 5.0, #
    #     'n_epochs': 100,
    #     'clip': 1.0,
    #     'lr': 1e-4,
    #     'discriminator_lr': 1e-5,
    #     'split_index': _SPLIT

    # }

    return Config(**kwargs)

# --------------------------------
# TRAIN
# --------------------------------
config = get_config(mode='train')
test_config = get_config(mode='test')

# print(config)
# print(test_config)
# print('split_index:', config.split_index)
print('Currently selected split_index:', config.split_index)
train_loader = get_loader(config.mode, config.split_index)
test_loader = get_loader(test_config.mode, test_config.split_index)
solver = Solver(config, train_loader, test_loader)

print('Training...')
TRAIN_TIME = time.time()

solver.build()
solver.evaluate(-1)	# evaluates the summaries generated using the initial random weights of the network 
solver.train()

elapsed_time = time.time() - TRAIN_TIME
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Train time: {hours} hrs, {minutes} mins, {seconds} s")

# --------------------------------
# GEN JSONS
# --------------------------------
def knapSack(W, wt, val, n): 
	K = [[0 for x in range(W + 1)] for x in range(n + 1)] 

	# Build table K[][] in bottom up manner 
	for i in range(n + 1): 
		for w in range(W + 1): 
			if i == 0 or w == 0:
				K[i][w] = 0 
			elif wt[i-1] <= w: 
				K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]) 
			else: 
				K[i][w] = K[i-1][w]

	selected = []
	w = W

	for i in range(n,0,-1):
		if K[i][w]!= K[i-1][w]:
			selected.insert(0,i-1)
			w -= wt[i-1]

	return selected 

def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions): 
    all_summaries = []
    for video_index in range(len(all_scores)):
    	# Get shots' boundaries
        shot_bound = all_shot_bound[video_index] # [number_of_shots, 2] - the boundaries refer to the initial number of frames (before the subsampling)
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        # Compute the importance scores for the initial frame sequence (not the subsampled one)
        frame_scores = np.zeros((n_frames), dtype=np.float32)
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
            shot_lengths.append(shot[1]-shot[0]+1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1]+1].mean()).item())
	
	# Select the best shots using the knapsack implementation
        final_max_length = int((shot[1]+1)*0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))
		
	# Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(shot[1]+1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1]+1] = 1
	
        all_summaries.append(summary)
		
    return all_summaries

def evaluate_summary(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary),user_summary.shape[1])
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
        if (precision+recall==0):
            f_scores.append(0)
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)
    

# path = 'exp0/fvs/results/split' + str(_SPLIT)  # path to the json files with the computed importance scores for each epoch
# PATH_TVSum = '../H5_file/fvs.h5' ### change to allinone.h5 for final dataset
path = f'{save_dir}/{config.video_type}/results/split{config.split_index}'
eval_method = 'avg' # the proposed evaluation method for TVSum videos

results = [f for f in listdir(path) if f.endswith(".json")]
# results.sort(key=lambda video: int(video[6:-5]))
results.sort(key=lambda ep: int(ep.split('_')[-1].split('.')[0]))
# DATASET_PATH= '../H5_file/fvs.h5'  #googlenet_2ps.h5' ### change to gn_full.h5 for oh dataset
dataset_path = f'../../datasets/{config.video_type}.h5'

# for each epoch, read the results' file and compute the f_score

# SUMMARY_DIR = 'exp0/fvs/summaries/split' + str(_SPLIT)
SUMMARY_DIR = save_dir.joinpath(config.video_type, 'summaries/split' + str(config.split_index))
os.makedirs(SUMMARY_DIR, exist_ok=True)

f_score_epochs = []
summaries_epoch = []
for epoch in tqdm(results):
    # print(epoch)
    all_scores = []
    with open(path+'/'+epoch) as f:
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:
            scores = np.asarray(data[video_name])
            all_scores.append(scores)

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
    with h5py.File(dataset_path, 'r') as hdf:        
        for video_name in keys:
            video_index = video_name[6:]
            
            user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
            sb = np.array( hdf.get('video_'+video_index+'/change_points') )
            n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
            positions = np.array( hdf.get('video_'+video_index+'/picks') )

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
    # print("f_score: ",np.mean(all_f_scores))
    # num_epoch = epoch.split("_")[1].split(".")[0]
    # with open(f'{SUMMARY_DIR}/summaries_{num_epoch}.pkl', 'wb') as file:
    #     pickle.dump(all_summaries, file)
    summaries_epoch.append(all_summaries)

f = max(f_score_epochs)   # returns the best f score
i = f_score_epochs.index(f)  # returns the best epoch number
best_summaries = summaries_epoch[i] # get summaries output for the best epoch
# save best summary
with open(f'{SUMMARY_DIR}/best_summaries.pkl', 'wb') as file:
    pickle.dump(best_summaries, file)

# print(f"BEST F-SCORE of {f} at EPOCH: {i}. Save file allinone_{i-1}.json")
# print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Summary: {SUMMARY_DIR}/summaries_{i}.pkl")
print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Best Summary: {SUMMARY_DIR}/best_summaries.pkl")
with open(path+'/f_scores.txt', 'w') as outfile:  
    json.dump(f_score_epochs, outfile)