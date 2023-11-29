import numpy as np
import math
import h5py
import json
import argparse
import pprint
import math
import sys
import os
from os import listdir
from tqdm import tqdm, trange
from pathlib import Path
from collections import Counter
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# from tensorboardX import SummaryWriter

# --------------------------------
# DATALOADER
# --------------------------------
def calculate_fragments(sequence_len, num_fragments):
    
    '''
    The sequence must be divided into "num_fragments" fragments.
    Since seq_len/num won't be a perfect division, we take both
    floor and ceiling parts, in a way such that the sum of all
    fragments will be equal to the total sequence.'''
    
    fragment_size = sequence_len/num_fragments
    fragment_floor = math.floor(fragment_size)
    fragment_ceil = math.ceil(fragment_size)
    i_part, d_part = divmod(fragment_size, 1)
    
    frag_jump = np.zeros(num_fragments)

    upper = d_part * num_fragments
    upper = np.round(upper).astype(int)
    lower = (1-d_part) * num_fragments
    lower = np.round(lower).astype(int)

    for i in range(lower):
        frag_jump[i] = fragment_floor
    for i in range(upper):
        frag_jump[lower+i] = fragment_ceil

    # Roll the scores, so that the larger fragments fall at 
    # the center of the sequence. Should not make a difference.
    frag_jump = np.roll(frag_jump, -int(num_fragments*(1-d_part)/2))

    if frag_jump[num_fragments-1] == 1:
        frag_jump[int(num_fragments/2)] = 1

    return frag_jump.astype(int)

def compute_fragments(seq_len, action_state_size):
    
    # "action_fragments" contains the starting and ending frame of each action fragment
    frag_jump = calculate_fragments(seq_len, action_state_size)
    action_fragments = torch.zeros((action_state_size,2), dtype=torch.int64)
    for i in range(action_state_size-1):
        action_fragments[i,1] = torch.tensor(sum(frag_jump[0:i+1])-1)
        action_fragments[i+1,0] = torch.tensor(sum(frag_jump[0:i+1]))
    action_fragments[action_state_size-1, 1] = torch.tensor(sum(frag_jump[0:action_state_size])-1)    
                
    return action_fragments

class VideoData(Dataset):
    def __init__(self, mode, split_index, action_state_size):
        self.mode = mode
        self.name = 'fvs'
        self.datasets = ['../../datasets/summe.h5',
                         '../../datasets/tvsum.h5',
                         '../../datasets/fvs.h5']
        # self.filename = '../H5_file/fvs.h5'
        # self.splits_filename = ['/content/drive/MyDrive/fair_vid_sum/splits/splits.json'] 
        # self.splits_filename = ['../splits/splits.json']
        self.splits_filename = ['../../splits/' + self.name + '_splits.json']
        self.split_index = split_index # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        elif 'fvs' in self.splits_filename[0]:
            self.filename = self.datasets[2]

        hdf = h5py.File(self.filename, 'r')
        self.action_fragments = {}
        self.list_features = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i==self.split_index:
                    self.split = split
                    
        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_features.append(features)
            self.action_fragments[video_name] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features and the action_fragments; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.split[self.mode + '_keys'][index]  #gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

def get_loader(mode, split_index, action_state_size):
    if mode.lower() == 'train':
        vd = VideoData(mode, split_index, action_state_size)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, split_index, action_state_size)
    


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

        # output: seq_len, batch, hidden_size * num_directions
        # h_n, c_n: num_layers * num_directions, batch_size, hidden_size
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
            prob: [batch_size, 1]
                Probability to be original feature from CNN
        """

        # [1, hidden_size]
        h = self.cLSTM(features)

        prob = self.out(h).squeeze()

        return h, prob
    
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
            features: [seq_len, 1, hidden_size] (compressed pool5 features)
        Return:
            scores: [seq_len, 1]
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
            last hidden:
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len: scalar (int)
            init_hidden:
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
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
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
            h: [2=num_layers, 1, hidden_size]
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
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """Actor that picks a fragment for the summary in every iteration"""
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, self.action_size)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            distribution: categorical distribution of pytorch
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """Critic that evaluates the Actor's choices"""
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            value: scalar
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        value = self.linear5(output)
        return value


# labels for training the GAN part of the model
original_label = torch.tensor(1.0).cuda()
summary_label = torch.tensor(0.0).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_returns(next_value, rewards, masks, gamma=0.99):
    """ Function that computes the return z_i following the equation (6) of the paper"""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates AC-SUM-GAN model"""
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
        self.actor = Actor(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size).cuda()
        self.critic = Critic(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator, self.actor, self.critic])

        if self.config.mode == 'train':
            # Build Optimizers
            self.e_optimizer = optim.Adam(
                self.summarizer.vae.e_lstm.parameters(),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                self.summarizer.vae.d_lstm.parameters(),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)
            self.optimizerA_s = optim.Adam(list(self.actor.parameters())
                                           + list(self.summarizer.s_lstm.parameters())
                                           + list(self.linear_compress.parameters()),
                                           lr=self.config.lr)
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.config.lr)

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

    def AC(self, original_features, seq_len, action_fragments):
        """ Function that makes the actor's actions, in the training steps where the actor and critic components are not trained"""
        scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

        fragment_scores = np.zeros(self.config.action_state_size)  # [num_fragments, 1]
        for fragment in range(self.config.action_state_size):
            fragment_scores[fragment] = scores[action_fragments[fragment,0]:action_fragments[fragment,1]+1].mean()
        state = fragment_scores

        previous_actions = []  # save all the actions (the selected fragments of each episode)
        reduction_factor = (self.config.action_state_size - self.config.termination_point) / self.config.action_state_size
        action_scores = (torch.ones(seq_len) * reduction_factor).cuda()
        action_fragment_scores = (torch.ones(self.config.action_state_size)).cuda()

        counter = 0
        for ACstep in range(self.config.termination_point):

            state = torch.FloatTensor(state).cuda()
            # select an action
            dist = self.actor(state)
            action = dist.sample()  # returns a scalar between 0-action_state_size

            if action not in previous_actions:
                previous_actions.append(action)
                action_factor = (self.config.termination_point - counter) / (self.config.action_state_size - counter) + 1

                action_scores[action_fragments[action, 0]:action_fragments[action, 1] + 1] = action_factor
                action_fragment_scores[action] = 0

                counter = counter + 1

            next_state = state * action_fragment_scores
            next_state = next_state.cpu().detach().numpy()
            state = next_state

        weighted_scores = action_scores.unsqueeze(1) * scores
        weighted_features = weighted_scores.view(-1, 1, 1) * original_features

        return weighted_features, weighted_scores

    def train(self):

        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            recon_loss_init_history = []
            recon_loss_history = []
            sparsity_loss_history = []
            prior_loss_history = []
            g_loss_history = []
            e_loss_history = []
            d_loss_history = []
            c_original_loss_history = []
            c_summary_loss_history = []
            actor_loss_history = []
            critic_loss_history = []
            reward_history = []            
            
            # Train in batches of as many videos as the batch_size
            num_batches = int(len(self.train_loader)/self.config.batch_size)
            iterator = iter(self.train_loader)
            for batch in range(num_batches):
                list_image_features = []
                list_action_fragments = []
                
                if self.config.verbose:
                    print(f'batch: {batch}')
                
                # ---- Train eLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training eLSTM...')
                self.e_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features, action_fragments = next(iterator)
                    
                    action_fragments = action_fragments.squeeze(0)
                    # [batch_size, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    
                    list_image_features.append(image_features)
                    list_action_fragments.append(action_fragments)
    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
    
                    weighted_features, scores = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    if self.config.verbose:
                        tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    prior_loss = self.prior_loss(h_mu, h_log_variance)
    
                    if self.config.verbose:
                        tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}')
    
                    e_loss = reconstruction_loss + prior_loss
                    e_loss = e_loss/self.config.batch_size
                    e_loss.backward()
                    
                    prior_loss_history.append(prior_loss.data)
                    e_loss_history.append(e_loss.data)
                    
                # Update e_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.e_lstm.parameters(), self.config.clip)
                self.e_optimizer.step()
                
                #---- Train dLSTM (decoder/generator) ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')
                self.d_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
    
                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    if self.config.verbose:
                        tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    g_loss = self.criterion(sum_prob, original_label)
                    
                    orig_features = original_features.squeeze(1)    # [seq_len, hidden_size]
                    gen_features = generated_features.squeeze(1)    #         >>
                    recon_losses = []
                    for frame_index in range(seq_len):
                        recon_losses.append(self.reconstruction_loss(orig_features[frame_index,:], gen_features[frame_index,:]))
                    reconstruction_loss_init = torch.stack(recon_losses).mean()

                    if self.config.verbose:
                        tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, g loss: {g_loss.item():.3f}')
                    
                    d_loss = reconstruction_loss + g_loss
                    d_loss = d_loss/self.config.batch_size
                    d_loss.backward()
                    
                    recon_loss_init_history.append(reconstruction_loss_init.data)
                    recon_loss_history.append(reconstruction_loss.data)
                    g_loss_history.append(g_loss.data)
                    d_loss_history.append(d_loss.data)
                    
                # Update d_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.d_lstm.parameters(), self.config.clip)
                self.d_optimizer.step()
                
                #---- Train cLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training cLSTM...')
                self.c_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # Train with original loss
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    h_origin, original_prob = self.discriminator(original_features)
                    c_original_loss = self.criterion(original_prob, original_label)
                    c_original_loss = c_original_loss/self.config.batch_size
                    c_original_loss.backward()
    
                    # Train with summary loss
                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
                    h_sum, sum_prob = self.discriminator(generated_features.detach())
                    c_summary_loss = self.criterion(sum_prob, summary_label)
                    c_summary_loss = c_summary_loss/self.config.batch_size
                    c_summary_loss.backward()
                    
                    if self.config.verbose:
                        tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
                    
                    c_original_loss_history.append(c_original_loss.data)
                    c_summary_loss_history.append(c_summary_loss.data)
                    
                # Update c_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(list(self.discriminator.parameters()) + list(self.linear_compress.parameters()), self.config.clip)
                self.c_optimizer.step()
                
                #---- Train sLSTM and actor-critic ----#
                if self.config.verbose:
                    tqdm.write('Training sLSTM, actor and critic...')
                self.optimizerA_s.zero_grad()
                self.optimizerC.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

                    fragment_scores = np.zeros(self.config.action_state_size)  # [num_fragments, 1]
                    for fragment in range(self.config.action_state_size):
                        fragment_scores[fragment] = scores[action_fragments[fragment, 0]:action_fragments[fragment, 1] + 1].mean()
    
                    state = fragment_scores  # [action_state_size, 1]
    
                    previous_actions = []  # save all the actions (the selected fragments of each step)
                    reduction_factor = (self.config.action_state_size - self.config.termination_point) / self.config.action_state_size
                    action_scores = (torch.ones(seq_len) * reduction_factor).cuda()
                    action_fragment_scores = (torch.ones(self.config.action_state_size)).cuda()
    
                    log_probs = []
                    values = []
                    rewards = []
                    masks = []
                    entropy = 0
    
                    counter = 0
                    for ACstep in range(self.config.termination_point):
                        # select an action, get a value for the current state
                        state = torch.FloatTensor(state).cuda()  # [action_state_size, 1]
                        dist, value = self.actor(state), self.critic(state)
                        action = dist.sample()  # returns a scalar between 0-action_state_size
    
                        if action in previous_actions:
    
                            reward = 0
    
                        else:
    
                            previous_actions.append(action)
                            action_factor = (self.config.termination_point - counter) / (self.config.action_state_size - counter) + 1
    
                            action_scores[action_fragments[action, 0]:action_fragments[action, 1] + 1] = action_factor
                            action_fragment_scores[action] = 0
    
                            weighted_scores = action_scores.unsqueeze(1) * scores
                            weighted_features = weighted_scores.view(-1, 1, 1) * original_features
    
                            h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                            h_origin, original_prob = self.discriminator(original_features)
                            h_sum, sum_prob = self.discriminator(generated_features)
    
                            if self.config.verbose:
                                tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                            rec_loss = self.reconstruction_loss(h_origin, h_sum)
                            reward = 1 - rec_loss.item()  # the less the distance, the higher the reward
                            counter = counter + 1
    
                        next_state = state * action_fragment_scores
                        next_state = next_state.cpu().detach().numpy()
    
                        log_prob = dist.log_prob(action).unsqueeze(0)
                        entropy += dist.entropy().mean()
    
                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
    
                        if ACstep == self.config.termination_point-1 :
                            masks.append(torch.tensor([0], dtype=torch.float, device=device)) 
                        else:
                            masks.append(torch.tensor([1], dtype=torch.float, device=device))
    
                        state = next_state
    
                    next_state = torch.FloatTensor(next_state).to(device)
                    next_value = self.critic(next_state)
                    returns = compute_returns(next_value, rewards, masks)
    
                    log_probs = torch.cat(log_probs)
                    returns = torch.cat(returns).detach()
                    values = torch.cat(values)
    
                    advantage = returns - values
    
                    actor_loss = -((log_probs * advantage.detach()).mean() + (self.config.entropy_coef/self.config.termination_point)*entropy)
                    sparsity_loss = self.sparsity_loss(scores)
                    critic_loss = advantage.pow(2).mean()
                    
                    actor_loss = actor_loss/self.config.batch_size
                    sparsity_loss = sparsity_loss/self.config.batch_size
                    critic_loss = critic_loss/self.config.batch_size
                    actor_loss.backward()
                    sparsity_loss.backward()
                    critic_loss.backward()
                    
                    reward_mean = torch.mean(torch.stack(rewards))
                    reward_history.append(reward_mean)
                    actor_loss_history.append(actor_loss)
                    sparsity_loss_history.append(sparsity_loss)
                    critic_loss_history.append(critic_loss)
                    
                    if self.config.verbose:
                        tqdm.write('Plotting...')
    
                    # self.writer.update_loss(original_prob.data, step, 'original_prob')
                    # self.writer.update_loss(sum_prob.data, step, 'sum_prob')
    
                    step += 1
                    
                # Update s_lstm, actor and critic parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.linear_compress.parameters())
                                           + list(self.summarizer.s_lstm.parameters())+list(self.critic.parameters()), self.config.clip)
                self.optimizerA_s.step()
                self.optimizerC.step()
                

            recon_loss_init = torch.stack(recon_loss_init_history).mean()
            recon_loss = torch.stack(recon_loss_history).mean()
            prior_loss = torch.stack(prior_loss_history).mean()
            g_loss = torch.stack(g_loss_history).mean()
            e_loss = torch.stack(e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_original_loss = torch.stack(c_original_loss_history).mean()
            c_summary_loss = torch.stack(c_summary_loss_history).mean()
            sparsity_loss = torch.stack(sparsity_loss_history).mean()
            actor_loss = torch.stack(actor_loss_history).mean()
            critic_loss = torch.stack(critic_loss_history).mean()
            reward = torch.mean(torch.stack(reward_history))

            # Plot
            # if self.config.verbose:
            #     tqdm.write('Plotting...')
            # self.writer.update_loss(recon_loss_init, epoch_i, 'recon_loss_init_epoch')
            # self.writer.update_loss(recon_loss, epoch_i, 'recon_loss_epoch')
            # self.writer.update_loss(prior_loss, epoch_i, 'prior_loss_epoch')    
            # self.writer.update_loss(g_loss, epoch_i, 'g_loss_epoch')    
            # self.writer.update_loss(e_loss, epoch_i, 'e_loss_epoch')
            # self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            # self.writer.update_loss(c_original_loss, epoch_i, 'c_original_loss_epoch')
            # self.writer.update_loss(c_summary_loss, epoch_i, 'c_summary_loss_epoch')
            # self.writer.update_loss(sparsity_loss, epoch_i, 'sparsity_loss_epoch')
            # self.writer.update_loss(actor_loss, epoch_i, 'actor_loss_epoch')
            # self.writer.update_loss(critic_loss, epoch_i, 'critic_loss_epoch')
            # self.writer.update_loss(reward, epoch_i, 'reward_epoch')

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

        for image_features, video_name, action_fragments in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, batch_size=1, input_size)]
            image_features = image_features.view(-1, self.config.input_size)
            image_features_ = Variable(image_features).cuda()

            # [seq_len, 1, hidden_size]
            original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
            seq_len = original_features.shape[0]
            
            with torch.no_grad():

                _, scores = self.AC(original_features, seq_len, action_fragments)

                scores = scores.squeeze(1)
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
# PARAMS
# --------------------------------

# save_dir = Path('AC-SUM-GAN/exp1')
exp_dir_name = 'exp2'
save_dir = Path(exp_dir_name)

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

        self.termination_point = math.floor(0.15*self.action_state_size)
        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        self.log_dir = save_dir.joinpath(video_type, 'logs/split' + str(self.split_index))
        self.score_dir = save_dir.joinpath(video_type, 'results/split' + str(self.split_index))
        self.save_dir = save_dir.joinpath(video_type, 'models/split' + str(self.split_index))
        
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
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--regularization_factor', type=float, default=5.0) # 0.5
    parser.add_argument('--entropy_coef', type=float, default=0.1)

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20) # 40
    parser.add_argument('--clip', type=float, default=1.0) # 5.0
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--action_state_size', type=int, default=8) # 60
    
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    # kwargs = {
    #     'mode' : 'train',
    #     'verbose' : False,
    #     'video_type' : 'fvs',
    #     'input_size' : 1024,
    #     'hidden_size' : 512,
    #     'num_layers' : 2,
    #     'regularization_factor': 5.0, #
    #     'entropy_coef': 0.1,
    #     'n_epochs': 100,
    #     'batch_size': 4,
    #     'clip': 1.0,
    #     'lr': 1e-4,
    #     'discriminator_lr': 1e-5,
    #     'split_index': _SPLIT,
    #     'action_state_size': 8 #

    # }

    # kwargs.update(optional_kwargs) #!

    return Config(**kwargs)

# --------------------------------
# TRAIN
# --------------------------------
config = get_config(mode='train')
test_config = get_config(mode='test')

# acs = 8
train_loader = get_loader(config.mode, config.split_index, config.action_state_size)
test_loader = get_loader(test_config.mode, test_config.split_index, test_config.action_state_size)
# train_loader = get_loader('train', _SPLIT, acs) # 0 is the split id, 30 is the action_state_size
# test_loader = get_loader('test', _SPLIT, acs)
solver = Solver(config, train_loader, test_loader)

print('Training...')
TRAIN_TIME = time.time()

solver.build()
solver.evaluate(-1)
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
    

# path = 'AC-SUM-GAN/exp1/fvs/results/split0'
# path = 'AC-SUM-GAN/exp1/fvs/results/split' + str(_SPLIT)
path = f'{save_dir}/{config.video_type}/results/split{config.split_index}'
dataset = config.video_type #'fvs' ### change to gn_full
eval_method = 'avg'

# results = listdir(path)
# results.sort(key=lambda video: int(video[6:-5]))

results = [f for f in listdir(path) if f.endswith(".json")]
# results.sort(key=lambda video: int(video[6:-5]))
results.sort(key=lambda ep: int(ep.split('_')[-1].split('.')[0]))
DATASET_PATH= f'../../datasets/{config.video_type}.h5' #'../H5_file/fvs.h5'   ### change to allinione.h5 for oh dataset

# summary_dir = 'AC-SUM-GAN/exp1/fvs/summaries/split' + str(_SPLIT)
# os.makedirs(summary_dir, exist_ok=True)
SUMMARY_DIR = save_dir.joinpath(config.video_type, 'summaries/split' + str(config.split_index))
os.makedirs(SUMMARY_DIR, exist_ok=True)


# for each epoch, read the results' file and compute the f_score
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
    with h5py.File(DATASET_PATH, 'r') as hdf:        
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
    # with open(f'{summary_dir}/summaries_{num_epoch}.pkl', 'wb') as file:
    #     pickle.dump(all_summaries, file)
    summaries_epoch.append(all_summaries)

f = max(f_score_epochs)   # returns the best f score
i = f_score_epochs.index(f)  # returns the best epoch number
best_summaries = summaries_epoch[i] # get summaries output for the best epoch
# save best summary
with open(f'{SUMMARY_DIR}/best_summaries.pkl', 'wb') as file:
    pickle.dump(best_summaries, file)

# print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file allinone_{i-1}")
# print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Summary: {summary_dir}/summaries_{i}.pkl")
print(f"BEST F-SCORE of {f:.2f} at EPOCH: {i}. Save file fvs_{i-1}.json, Best Summary: {SUMMARY_DIR}/best_summaries.pkl")
with open(path+'/f_scores.txt', 'w') as outfile:  
    json.dump(f_score_epochs, outfile)