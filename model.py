import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
import math
from kmeans import cached_kmeans
import cProfile as profile

from timeit import timeit

import numpy as np

class FixupResNetCNN(nn.Module):
    """source: https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py"""

    class _FixupResidual(nn.Module):
        def __init__(self, depth, num_residual):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for p in self.conv1.parameters():
                p.data.mul_(1 / math.sqrt(num_residual))
            for p in self.conv2.parameters():
                p.data.zero_()
            self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

        def forward(self, x):
            print('REsInput: ', x.shape)
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4
            print('REsOutput: ', out.shape)
            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        depth_in = input_channels

        layers = []
        if not double_channels:
            channel_sizes = [32, 64, 64]
        else:
            channel_sizes = [64, 128, 128]
        for depth_out in channel_sizes:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._FixupResidual(depth_out, 8),
                self._FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            self._FixupResidual(depth_in, 8),
            self._FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        print('Core_Input',x.shape)
        return self.conv_layers(x)


class InputProcessor(nn.Module):

# TODO make this work for arbitrary nonspati shape.
# Use Extra None Case. Use (1024-nonspatialsize) for spatial out.

    def __init__(self):
        super().__init__()
        self.conv_layers = FixupResNetCNN(3,double_channels=True)
        self.spatial_reshape = nn.Sequential(nn.Linear(128*8*8, 1024),nn.ReLU(),nn.LayerNorm(1024))
        #self.nonspatial_reshape = nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.LayerNorm(128))

    def forward(self, spatial, nonspatial):
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        print('pov before Core:', spatial.shape)
        spatial = self.conv_layers(spatial)
        print('pov after Core:', spatial.shape)
        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        print('pov after reshape:', spatial.shape)
        print('nonspatial before Core:', nonspatial.shape)
        #nonspatial = self.nonspatial_reshape(nonspatial)
        print('nonspatial after FC:', nonspatial.shape)
        spatial = self.spatial_reshape(spatial)
        print('spatial after FC:', spatial.shape)

        print('Core_out: ', torch.cat([spatial, nonspatial],dim=-1).shape)

        return spatial


class Core(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.lstm = nn.LSTM(1024, 1024, 1)
        
    def forward(self, spatial, nonspatial, state):

        print("starting_timeit")



#        print('time_CNN:',timeit("processed = input_proc.forward(spatial, nonspatial)",number = 10,globals=locals()))



        processed = self.input_proc.forward(spatial, nonspatial)

        #print('time_lstm:',timeit("lstm_output, new_state = lstm(processed, state)",number = 10,globals=locals()))

        #print("finished_timeit")

        print('State0: ',state[0].shape)
        print('State1: ',state[0].shape)

        lstm_output, new_state = self.lstm(processed, state)
        print('lstm_out:',lstm_output.shape)
        print('Core_out_total:',(lstm_output+processed).shape)
        return lstm_output+processed, new_state


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.kmeans = cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
        self.core = Core()
        self.selector = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 120))

    def get_zero_state(self, batch_size, device="cpu"):
        return (torch.zeros((1, batch_size, 1024), device=device), torch.zeros((1, batch_size, 1024), device=device))

    def compute_front(self, spatial, nonspatial, state):
        hidden, new_state = self.core(spatial, nonspatial, state)
        #print('after core: hidden,new_state  = ',hidden[0].shape,new_state.shape)
        print('after_selector : hidden state',self.selector(hidden)[0].shape)
        return hidden, self.selector(hidden), new_state

    def forward(self, spatial, nonspatial, state, target):
        pass

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point):

        point = point.long()

        loss = nn.CrossEntropyLoss()
        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        print('d shape: ',d.shape)
        print('point shape: ',point.shape)
        print('d_view_shape: ',d.view(-1, d.shape[-1]).shape)
        print('point_view_shape: ',point.view(-1).shape)

        print(d.shape)
        print()
        l1 = loss(d.view(-1, d.shape[-1]), point.view(-1))
        print('l1 shape: ', l1.item())
        return l1, {"action":l1.item()}, state

    def sample(self, spatial, nonspatial, prev_action, state, target):
        print('pov_input = ',spatial.shape)
        print('obfs_input = ',nonspatial.shape)
        print('hidden_states = ',state[0].shape)

        print(self.core)

        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        dist = D.Categorical(logits = d)
        s = dist.sample()
        s = s.squeeze().cpu().numpy()
        return self.kmeans.cluster_centers_[s], state
