import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
import math
from kmeans import cached_kmeans
import cProfile as profile

from timeit import timeit

import numpy as np
verb = False

def verb_print(String):
    if verb:
        print(String)

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
            verb_print('REsInput: ', x.shape)
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4
            verb_print('REsOutput: ', out.shape)
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
        verb_print('Core_Input',x.shape)
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
        verb_print('pov before Core:', spatial.shape)
        spatial = self.conv_layers(spatial)
        verb_print('pov after Core:', spatial.shape)
        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        verb_print('pov after reshape:', spatial.shape)
        verb_print('nonspatial before Core:', nonspatial.shape)
        #nonspatial = self.nonspatial_reshape(nonspatial)
        verb_print('nonspatial after FC:', nonspatial.shape)
        spatial = self.spatial_reshape(spatial)
        verb_print('spatial after FC:', spatial.shape)

        verb_print('Core_out: ', torch.cat([spatial, nonspatial],dim=-1).shape)

        return spatial


class Core(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.lstm = nn.LSTM(1024, 1024, 1)
        
    def forward(self, spatial, nonspatial, state):

        verb_print("starting_timeit")



#        verb_print('time_CNN:',timeit("processed = input_proc.forward(spatial, nonspatial)",number = 10,globals=locals()))



        processed = self.input_proc.forward(spatial, nonspatial)

        #verb_print('time_lstm:',timeit("lstm_output, new_state = lstm(processed, state)",number = 10,globals=locals()))

        #print("finished_timeit")

        verb_print('State0: ',state[0].shape)
        verb_print('State1: ',state[0].shape)

        lstm_output, new_state = self.lstm(processed, state)
        verb_print('lstm_out:',lstm_output.shape)
        verb_print('Core_out_total:',(lstm_output+processed).shape)
        return lstm_output+processed, new_state



class Model(nn.Module):

    def __init__(self,verbose=False):
        super().__init__()
        self.kmeans = cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
        self.core = Core()
        self.selector = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 120))
        global verb
        verb = verbose

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 1024), device=device), torch.zeros((1, batch_size, 1024), device=device))

    def compute_front(self, spatial, nonspatial, state):
        hidden, new_state = self.core(spatial, nonspatial, state)
        #verb_print('after core: hidden,new_state  = ',hidden[0].shape,new_state.shape)
        verb_print('after_selector : hidden state',self.selector(hidden)[0].shape)
        return hidden, self.selector(hidden), new_state

    def forward(self, spatial, nonspatial, state, target):
        pass

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point):

        point = point.long()

        loss = nn.CrossEntropyLoss()
        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        verb_print('d shape: ',d.shape)
        verb_print('point shape: ',point.shape)
        verb_print('d_view_shape: ',d.view(-1, d.shape[-1]).shape)
        verb_print('point_view_shape: ',point.view(-1).shape)

        verb_print(d.shape)
        verb_print()
        l1 = loss(d.view(-1, d.shape[-1]), point.view(-1))
        verb_print('l1 shape: ', l1.item())
        return l1, {"action":l1.item()}, state

    def sample(self, spatial, nonspatial, prev_action, state, target):
        verb_print('pov_input = ',spatial.shape)
        verb_print('obfs_input = ',nonspatial.shape)
        verb_print('hidden_states = ',state[0].shape)

        verb_print(self.core)

        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        dist = D.Categorical(logits = d)
        s = dist.sample()
        s = s.squeeze().cuda().numpy()
        return self.kmeans.cluster_centers_[s], state
