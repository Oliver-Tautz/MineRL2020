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
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional

import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='train the model ...')
parser.add_argument('modelname', help="name of the model", type=str)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--map-to-zero', help="map non recorded actions to zero", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--c', help="make torch use number of cpus", type=int, default=4)
parser.add_argument('--epochs', help="make torch use number of cpus", type=int, default=100)
parser.add_argument('--batchsize', help="make torch use number of cpus", type=int, default=4)
parser.add_argument('--seq-len', help="make torch use number of cpus", type=int, default=100)
parser.add_argument('--no-classes', help="use number of discrete actions", type=int, default=30)
parser.add_argument('--max-overlap', help="max overlap of seueqnces in dataset in frames", type=int, default=10)
parser.add_argument('--debug', help="use small number of samples for debugging faster", action='store_true')
parser.add_argument('--no-shuffle', help="dont shuffle train set after each epoch", action='store_true')
parser.add_argument('--no-sequences', help="use number of sequences for train/val dataset", type=int, default=5000)
parser.add_argument('--val-split', help="split into val set. ", type=float, default=0.2)
parser.add_argument('--min-reward', help="min reward per sequence", type=int, default=0)
parser.add_argument('--min-var', help="min action variance in sequence", type=int, default=0)
parser.add_argument('--loss-position', help="only predict one label at position in sequence. Set to -1 to predict all positions.", type=int, default=-1)

parser.add_argument('--weight_loss', help="wheight loss for under/overrepresented classes", action="store_true")

args = parser.parse_args()

# there are 448480 unique steps in the dataset.
# so choose > 450000/seq len sequences ...
no_sequences = args.no_sequences
val_split = args.val_split
no_shuffle = args.no_shuffle
modelname = args.modelname
map_to_zero = args.map_to_zero
with_masks = args.with_masks
verb = args.verbose
epochs = args.epochs
no_classes = args.no_classes
min_reward = args.min_reward
min_var = args.min_var
max_overlap = args.max_overlap
loss_position = args.loss_position
weight_loss = args.weight_loss
class_weights = None

# ensure reproducability

torch.manual_seed(12)
random.seed(12)
np.random.seed(12)

model_index = 0

while (True):
    try:
        os.makedirs(f"train/{modelname}_{model_index}", exist_ok=False)
        model_folder = f"train/{modelname}_{model_index}"
        modelname= f"{modelname}_{model_index}"
        break
    except:
        model_index += 1


def verb_print(*strings):
    global verb
    if verb:
        print(strings)

torch.set_num_threads(8)
deviceStr = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('using device:', device, file=sys.stderr)

from random import shuffle

from minerl.data import DataPipeline

batchsize = args.batchsize
seq_len = args.seq_len

if args.debug:
    no_replays = 2
    epochs = 5
    no_sequences = 40
else:
    no_replays = 300


def train(model, epochs, train_loader, val_loader):
    torch.set_num_threads(args.c)
    optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1) * (sqrt(sqrt(sqrt(10))) ** min(x, 50)), 1)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()

    gradsum = 0

    timestring = "%Y-%m-%d-%H-%M-%S"
    simple_logger = SimpleLogger(
        f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={seq_len}_time={datetime.now().strftime(timestring)}.csv",
        ['modelname','epoch', 'loss', 'val_loss', 'grad_norm', 'learning_rate', 'seq_len', 'map_to_zero', 'batchsize', 'no_classes',
         'no_sequences','min_reward','min_var','with_masks','max_overlap'])

    additional_info_dummy = torch.zeros(10)
    best_val_loss = 1000
    lstm_state = model.get_zero_state(batchsize)
    for epoch in trange(epochs, desc='epochs'):

        model.train()
        # save batch losses
        epoch_train_loss = []
        epoch_val_loss = []

        # train on batches
        for pov, act in tqdm(train_loader, desc='batch_train', position=0, leave=True):
            # reset lstm_state after each sequence
            #lstm_state = model.get_zero_state(batchsize)

            # swap batch and seq; swap x and c; swap x and y back. Is this necessary? Be careful in testing! match this operation.
            pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

            # move to gpu if not there
            if not pov.is_cuda or not act.is_cuda:
                pov, act = pov.to(deviceStr), act.to(deviceStr)
            else:
                pass

            # loss, ldict, lstm_state = model.get_loss(pov, additional_info_dummy, additional_info_dummy, lstm_state,
            #                                    torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

            prediciton, _ = model.forward(pov, additional_info_dummy, lstm_state)

            if loss_position < 0:

                loss = categorical_loss(act, prediciton)
            else:
                loss = categorical_loss_one_action(act,prediciton,loss_position)


            loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss.append(loss.item())

        ### Eval  #####

        with torch.no_grad():
            model.eval()

            for pov, act in tqdm(val_loader, desc='batch_eval', position=0, leave=True):
                # reset lstm_state
                #lstm_state = model.get_zero_state(batchsize)
                pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

                # move to gpu
                pov, act = pov.to(deviceStr), act.to(deviceStr)

                # move to gpu if not there
                if not pov.is_cuda or not act.is_cuda:
                    pov, act = pov.to(deviceStr), act.to(deviceStr)
                else:
                    pass#print('this is actually useful maybe?')

                prediciton, _ = model.forward(pov, additional_info_dummy, lstm_state)

                val_loss = categorical_loss(act, prediciton)


                epoch_val_loss.append(val_loss.item())

            if (epoch % 4) == 0:
                print("------------------Saving Model!-----------------------")
                torch.save(model.state_dict(),
                           f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={seq_len}_epoch={epoch}_time={datetime.now()}.tm")

            elif (sum(epoch_val_loss) / len(epoch_val_loss)) < best_val_loss:
                best_val_loss = (sum(epoch_val_loss) / len(epoch_val_loss))
                torch.save(model.state_dict(),
                           f"{model_folder}/{modelname}_with-masks={with_masks}_map-to-zero={map_to_zero}_no-classes={no_classes}_seq-len={seq_len}_epoch={epoch}_time={datetime.now()}.tm")

            print("-------------Logging!!!-------------")
            print(f"current loss = {sum(epoch_train_loss) / len(epoch_train_loss)}; val_loss={sum(epoch_val_loss) / len(epoch_val_loss)}")
            simple_logger.log(
                [modelname,epoch, sum(epoch_train_loss) / len(epoch_train_loss), sum(epoch_val_loss) / len(epoch_val_loss),
                 gradsum, float(optimizer.param_groups[0]["lr"]), seq_len, map_to_zero, batchsize, no_classes,
                 no_sequences,min_reward,min_var,with_masks,max_overlap])

            gradsum = 0
            scheduler.step()


# drift : negative drift for drifting backwards in time, positive drift for drifting forward.
def drift_loss(prediction, label, drift=1):
    if drift > 0:
        prediction = prediction[:-drift]
        label = label[:, drift:]
    elif drift < 0:
        drift = abs(drift)
        prediction = prediction[drift:]
        label = label[:, :-drift]
    else:
        # here we compute normal loss
        pass

    return categorical_loss(label, prediction)


# label     : label with shape (batch,sequence)
# prediction: prediction with shape (sequence,batch,no_actions)

def categorical_loss(label, prediction):
    global class_weights

    if class_weights == None:
        class_weights=torch.ones(no_classes)

    label = label.transpose(0, 1)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # flatten

    label = label.reshape(-1)
    prediction = prediction.view(-1, prediction.shape[-1])

    # normalize by batchsize and seq len.
    return (loss(prediction, label)/len(prediction)).sum()

def categorical_loss_one_action(label,prediction,position):
    loss = torch.nn.BCEWithLogitsLoss()

    #label = label.reshape(-1)
    prediction = prediction.transpose(0,1)

    for batch_label, batch_pred in zip(label,prediction):
        print(batch_pred.shape,batch_pred.shape)

    label_pos = label[position]
    prediction_pos = prediction[position]
    print(label_pos,prediction_pos)

    return torch.nn.BCEWithLogitsLoss()(prediction_pos,label_pos)


def compute_class_weights(mineDs):
    acts = []
    for pov, act in mineDs:
        acts.append(act)

    acts = np.concatenate(acts)
    _, counts = np.unique(acts, return_counts=True)

    # weight
    weights = counts/sum(counts)
    weights = torch.tensor(weights,device=deviceStr,dtype=torch.float32)
    weights=1/weights

    return  torch.nn.functional.pad(weights,(0,no_classes-len(weights)),mode='constant',value=1)


def main():
    global no_sequences
    model = Model(deviceStr=deviceStr, verbose=False, no_classes=no_classes, with_masks=with_masks)

    os.makedirs("train", exist_ok=True)

    full_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=seq_len, map_to_zero=map_to_zero,
                           with_masks=with_masks, no_classes=no_classes, no_replays=no_replays,
                           random_sequences=no_sequences, min_variance=min_var, min_reward=min_reward, device=deviceStr,max_overlap=max_overlap)

    no_sequences = len(full_set)
    val_split = 0.2

    train_size = int(no_sequences * (1 - val_split))
    val_size = int(no_sequences * val_split)

    if weight_loss:
        global class_weights
        class_weights = compute_class_weights(full_set)

    if train_size+val_size< no_sequences:
        train_size+=1

    train_set, val_set = random_split(full_set, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    if no_shuffle:
        train_loader = DataLoader(train_set, batch_size=batchsize,
                                  shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    else:
        train_loader = DataLoader(train_set, batch_size=batchsize,
                                  shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=batchsize,
                            shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    print(f"training on {len(train_set)} sequences of length {seq_len}, validation on {len(val_set)}")

    if deviceStr == "cuda":
        model.cuda()
    else:
        model.cpu()

    # model.load_state_dict(torch.load(f"train/trained_models/first_run/model_14.tm", map_location=device))
    print(
        'Starting training with map_to_zero={}, modelname={}, with_masks={}, no_actions={}'.format(map_to_zero,
                                                                                                   modelname,
                                                                                                   with_masks,
                                                                                                   no_classes))

    train(model, epochs, train_loader, val_loader)

    print('training done!')


if __name__ == "__main__":
    main()
