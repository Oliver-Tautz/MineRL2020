# TODO

import time
import logging
import os
import sys
import argparse

from tqdm import tqdm

import coloredlogs

# coloredlogs.install(logging.DEBUG)

from model import Model
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from time import time
from loader import BatchSeqLoader, absolute_file_paths
from math import sqrt
from kmeans import cached_kmeans
from simple_logger import SimpleLogger
import numpy as np
import random
from mineDataset import MineDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='train the model ...')
parser.add_argument('modelname', help="name of the model", type=str)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--map-to-zero', help="map non recorded actions to zero", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--c', help="make torch use number of cpus", default=12)
parser.add_argument('--epochs', help="make torch use number of cpus", default=100)
parser.add_argument('--batchsize', help="make torch use number of cpus", default=4)
parser.add_argument('--seq-len', help="make torch use number of cpus", default=100)

args = parser.parse_args()

# In ONLINE=True mode the code saves only the final version with early stopping,
# in ONLINE=False it saves 20 intermediate versions during training.
ONLINE = True

trains_loaded = True

number_of_checkpoints = 20
modelname = args.modelname

# ensure reproducability

torch.manual_seed(12)
random.seed(12)
np.random.seed(12)

try:
    os.makedirs("train/{}".format(modelname), exist_ok=True)
except:
    print("Model already present!")
    exit()

map_to_zero = args.map_to_zero
with_masks = args.with_masks
verb = args.verbose
epochs = args.epochs

print(verb)


def verb_print(*strings):
    global verb
    if verb:
        print(strings)


deviceStr = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    from clearml import Task
except:
    trains_loaded = False
from random import shuffle

from minerl.data import DataPipeline

BATCH_SIZE = args.batchsize
SEQ_LEN = args.seq_len


def train(model, epochs, train_loader, val_loader):
    torch.set_num_threads(args.c)
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1) * (sqrt(sqrt(sqrt(10))) ** min(x, 50)), 1)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()
    step = 0

    gradsum = 0

    simple_logger = SimpleLogger("loss_csv/{}.csv".format(modelname),
                                 ['epoch', 'loss', 'val_loss', 'grad_norm', 'learning_rate'])

    nonspatial_dummy = torch.zeros(10)

    for epoch in tqdm(range(epochs), desc='epochs'):

        # save batch losses
        epoch_train_loss = []
        epoch_val_loss = []

        # train on batches
        for pov, act in tqdm(train_loader, desc='batch_train'):
            # reset hidden
            hidden = model.get_zero_state(BATCH_SIZE)

            # swap batch and seq; swap x and c; swap x and y back. Is this necessary? Be careful in testing! match this operation.
            pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4)

            # move to gpu
            pov.to(deviceStr)
            act.to(deviceStr)

            loss, ldict, hidden = model.get_loss(pov, nonspatial_dummy, nonspatial_dummy, hidden,
                                                 torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

            loss = loss.sum()
            loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_train_loss.append(loss.item())

        ### Eval  #####

        with torch.no_grad():
            model.eval()

            for pov, act in tqdm(train_loader, desc='batch_eval'):
                # reset hidden
                hidden = model.get_zero_state(BATCH_SIZE)
                pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4)

                # move to gpu
                pov.to(deviceStr)
                act.to(deviceStr)

                loss, ldict, hidden = model.get_loss(pov, nonspatial_dummy, nonspatial_dummy, hidden,
                                                     torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

                loss = loss.sum()
                epoch_val_loss.append(loss.item())

                print("------------------Saving Model!-----------------------")
            torch.save(model.state_dict(), f"train/{modelname}/{modelname}_{epoch}.tm")
            torch.save(model.state_dict(), f"train/{modelname}/{modelname}.tm")

            print("-------------Logging!!!-------------")
            simple_logger.log(
                [epoch, sum(epoch_train_loss) / len(epoch_train_loss), sum(epoch_val_loss) / len(epoch_val_loss),
                 gradsum, float(optimizer.param_groups[0]["lr"])])

            gradsum = 0


def main():
    # a bit of code that creates clearml logging (formerly trains) if clearml
    # is available

    model = Model(deviceStr=deviceStr, verbose=True, no_classes=30, with_masks=with_masks)

    os.makedirs("train", exist_ok=True)

    train_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=SEQ_LEN, map_to_zero=map_to_zero,
                            with_masks=with_masks, no_replays=3)

    val_set = MineDataset('data/MineRLTreechop-v0/val', sequence_length=SEQ_LEN, map_to_zero=map_to_zero,
                          with_masks=with_masks, no_replays=1)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, drop_last=True)
    # print(spatial.shape)
    # print(nonspatial.shape)
    # print(prev_action.shape)
    # print(act.shape)
    # print(hidden.shape)

    # summary(Model)

    # if LOAD:
    #    model.load_state_dict(torch.load("train/some_model.tm"))
    if deviceStr == "cuda":
        model.cuda()
    else:
        model.cpu()

    # model.load_state_dict(torch.load(f"train/trained_models/first_run/model_14.tm", map_location=device))
    print(
        'Starting training with map_to_zero={}, modelname={}, with_masks={}'.format(map_to_zero, modelname, with_masks))

    train(model, 3, train_loader, val_loader)

    print('training done!')
    torch.save(model.state_dict(), "train/some_model.tm")
    print("ok", file=sys.stderr)


if __name__ == "__main__":
    main()
