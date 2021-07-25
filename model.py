import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sigmoid
import torch.distributions as D
import math
from kmeans import cached_kmeans
import use_resnet


from timeit import timeit

import numpy as np
verb = False

def verb_print(*strings):
    if verb:
        print(strings)

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

    def __init__(self,input_channels):
        super().__init__()
        self.conv_layers = FixupResNetCNN(input_channels,double_channels=True)
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
        #verb_print('nonspatial before Core:', nonspatial.shape)
        #nonspatial = self.nonspatial_reshape(nonspatial)
        #verb_print('nonspatial after FC:', nonspatial.shape)
        spatial = self.spatial_reshape(spatial)
        verb_print('spatial after FC:', spatial.shape)

        #verb_print('Core_out: ', torch.cat([spatial, nonspatial],dim=-1).shape)

        return spatial




class Model(nn.Module):


    # verbose    : print lots of  stuff
    # deviceStr  : use device for model functions
    # no_classes : number of classes to predict
    # with_masks : is a mask channel (c=4) supplied?
    # mode       : if mode = train: don't compute masks , else : do it!

    def __init__(self, verbose=False, deviceStr='cuda',no_classes=30,with_masks = False,mode='train',with_lstm = True):
        super().__init__()
        self.mode = mode

        ## init model

        if with_masks:
            self.cnn_head = InputProcessor(input_channels=4)
        else:
            self.cnn_head = InputProcessor(input_channels=3)

        if with_lstm:
            self.lstm = nn.LSTM(1024,1024,1)

        # Dont use Softmax here! Its applied by nn.CrossEntropyLoss().
        self.no_classes=no_classes
        self.selector = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024,self.no_classes))
        global verb
        verb = verbose
        self.deviceStr=deviceStr
        self.with_masks = with_masks
        self.with_lstm = with_lstm

        if with_masks and mode != 'train':
            self.masksGenerator = use_resnet.MaskGeneratorResnet(self.deviceStr)

        self.logits_mean = nn.Parameter(torch.zeros(no_classes),requires_grad=False)
        self.logits_sdt =  nn.Parameter(torch.zeros(no_classes),requires_grad=False)



    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 1024), device=self.deviceStr), torch.zeros((1, batch_size, 1024), device=self.deviceStr))

    def set_logits_mean_and_std(self,mean,std):
        self.logits_mean =  nn.Parameter(mean,requires_grad=False)
        self.logits_sdt =  nn.Parameter(std,requires_grad=False)


    def sample(self, spatial, nonspatial, state,mean_substract=False):

        logits,state =  self.forward(spatial,nonspatial,state)

        pred = torch.argmax((torch.nn.Softmax(dim=0)(logits.view(-1))))

        if mean_substract:
        # intialized with zeros, so always ok ...
            logits = logits-self.logits_mean

        probs = Sigmoid()(logits)
        dist = D.Categorical(probs = probs)
        sampled_pred = dist.sample()
        sampled_pred = sampled_pred.squeeze().cpu().numpy()

        return sampled_pred, pred, state

    def sample_multilabel(self,spatial, nonspatial, state,mean_substract=False):

        logits, state = self.forward(spatial, nonspatial, state)

        probs = Sigmoid()(logits)
        pred = probs>0.5

        if mean_substract:
        # intialized with zeros, so always ok ...
            logits = logits-self.logits_mean



        dist = D.bernoulli.Bernoulli(probs=probs)
        sampled_pred = dist.sample()
        sampled_pred = sampled_pred.squeeze().cpu().numpy()

        return sampled_pred, pred, state, logits


    def forward(self,pov,additional_info,state):

        if self.with_masks and self.mode != 'train':
            # spatialnp = spatial.numpy()
            sequence, batch, c, x, y = pov.shape
            pov = self.masksGenerator.append_channel_in_model(pov)


        latent_pov = self.cnn_head(pov,additional_info)


        if self.with_lstm:
            out, new_state = self.lstm(latent_pov,state)
        else:

            out = latent_pov

        out = self.selector(out)

        return out, state
