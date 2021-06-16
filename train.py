

import time
import logging
import os
import sys
import argparse

from tqdm import tqdm, trange

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
from datetime import datetime

parser = argparse.ArgumentParser(description='train the model ...')
parser.add_argument('modelname', help="name of the model", type=str)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--map-to-zero', help="map non recorded actions to zero", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--c', help="make torch use number of cpus",type=int, default=4)
parser.add_argument('--epochs', help="make torch use number of cpus", type=int,default=100)
parser.add_argument('--batchsize', help="make torch use number of cpus",type=int, default=4)
parser.add_argument('--seq-len', help="make torch use number of cpus",type=int, default=100)
parser.add_argument('--no-classes', help="use number of distcrete actions",type=int, default=30)
parser.add_argument('--debug', help="use small number of samples for debugging faster",action='store_true')

args = parser.parse_args()

# In ONLINE=True mode the code saves only the final version with early stopping,
# in ONLINE=False it saves 20 intermediate versions during training.
ONLINE = True

trains_loaded = True

modelname = args.modelname

# ensure reproducability

torch.manual_seed(12)
random.seed(12)
np.random.seed(12)

model_index = 0

while (True):
    try:
        os.makedirs(f"train/{modelname}_{model_index}", exist_ok=False)
        model_folder = f"train/{modelname}_{model_index}"
        break
    except:
        model_index+=1


map_to_zero = args.map_to_zero
with_masks = args.with_masks
verb = args.verbose
epochs = args.epochs
no_classes = args.no_classes


def verb_print(*strings):
    global verb
    if verb:
        print(strings)


deviceStr = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('using device:', device,file=sys.stderr)

try:
    from clearml import Task
except:
    trains_loaded = False
from random import shuffle

from minerl.data import DataPipeline

BATCH_SIZE = args.batchsize
SEQ_LEN = args.seq_len

if args.debug:
    no_replays = 2
else:
    no_replays =300


def train(model, epochs, train_loader, val_loader):

    torch.set_num_threads(args.c)
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1) * (sqrt(sqrt(sqrt(10))) ** min(x, 50)), 1)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()


    gradsum = 0

    simple_logger = SimpleLogger(f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={SEQ_LEN}_time={datetime.now()}.csv",
                                 ['epoch', 'loss', 'val_loss', 'grad_norm', 'learning_rate'])

    nonspatial_dummy = torch.zeros(10)
    best_val_loss = 1000

    for epoch in trange(epochs, desc='epochs'):
        model.train()
        # save batch losses
        epoch_train_loss = []
        epoch_val_loss = []

        # train on batches
        for pov, act in tqdm(train_loader, desc='batch_train',position=0,leave=True):
            # reset hidden
            hidden = model.get_zero_state(BATCH_SIZE)

            # swap batch and seq; swap x and c; swap x and y back. Is this necessary? Be careful in testing! match this operation.
            pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

            # move to gpu if not there
            if not pov.is_cuda or not act.is_cuda:
                pov, act = pov.to(deviceStr), act.to(deviceStr)

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

            for pov, act in tqdm(val_loader, desc='batch_eval',position=0,leave=True):
                # reset hidden
                hidden = model.get_zero_state(BATCH_SIZE)
                pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

                # move to gpu
                pov, act = pov.to(deviceStr), act.to(deviceStr)

                # move to gpu if not there
                if not pov.is_cuda or not act.is_cuda:
                    pov, act = pov.to(deviceStr), act.to(deviceStr)
                else:
                    print('this is actually useful maybe?')

                val_loss, val_ldict, hidden = model.get_loss(pov, nonspatial_dummy, nonspatial_dummy, hidden,
                                                     torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

                val_loss = val_loss.sum()
                epoch_val_loss.append(val_loss.item())

            if (epoch%5) == 0:
                print("------------------Saving Model!-----------------------")
                torch.save(model.state_dict(), f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={SEQ_LEN}_epoch={epoch}_time={datetime.now()}.tm")

            if (sum(epoch_train_loss) / len(epoch_train_loss)) < best_val_loss:
                best_val_loss = (sum(epoch_train_loss) / len(epoch_train_loss))
                torch.save(model.state_dict(),f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={SEQ_LEN}_epoch={epoch}_time={datetime.now()}.tm")

            print("-------------Logging!!!-------------")
            simple_logger.log(
                [epoch, sum(epoch_train_loss) / len(epoch_train_loss), sum(epoch_val_loss) / len(epoch_val_loss),
                 gradsum, float(optimizer.param_groups[0]["lr"])])

            gradsum = 0


def main():

    model = Model(deviceStr=deviceStr, verbose=False, no_classes=no_classes, with_masks=with_masks)

    os.makedirs("train", exist_ok=True)

    train_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=SEQ_LEN, map_to_zero=map_to_zero,
                            with_masks=with_masks,no_classes=no_classes,no_replays=no_replays)

    val_set = MineDataset('data/MineRLTreechop-v0/val', sequence_length=SEQ_LEN, map_to_zero=map_to_zero,
                          with_masks=with_masks,no_classes=no_classes,no_replays=no_replays)


    # shuffle only train set.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, drop_last=True,pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, drop_last=True,pin_memory=True)



    if deviceStr == "cuda":
        model.cuda()
    else:
        model.cpu()

    # model.load_state_dict(torch.load(f"train/trained_models/first_run/model_14.tm", map_location=device))
    print(
        'Starting training with map_to_zero={}, modelname={}, with_masks={}, no_actions={}'.format(map_to_zero, modelname, with_masks,no_classes))

    train(model, epochs, train_loader, val_loader)

    print('training done!')
    print("ok", file=sys.stderr)


if __name__ == "__main__":
    main()
