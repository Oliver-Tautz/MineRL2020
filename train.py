# Simple env test.
import json
import select
import time
import logging
import os
import sys
import argparse

from tqdm import tqdm

import coloredlogs



#coloredlogs.install(logging.DEBUG)

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

import argparse

parser = argparse.ArgumentParser(description='train the model ...')
parser.add_argument('modelname', help="name of the model", type=str)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--map-to-zero', help="map non recorded actions to zero", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--c', help="make torch use number of cpus", default=12)

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
    os.makedirs("train/{}".format(modelname), exist_ok=False)
except:
    print("Model already present!")
    exit()

map_to_zero = args.map_to_zero
with_masks = args.with_masks
verb = args.verbose


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

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
print(MINERL_DATA_ROOT, file=sys.stderr)

BATCH_SIZE = 4
SEQ_LEN = 100

FIT = True
LOAD = False
FULL = True


def update_loss_dict(old, new):
    if old is not None:
        for k in old:
            old[k] += new[k]
        return old
    return new


def train(model, mode, steps, train_loader, val_loader, logger):
    torch.set_num_threads(args.c)
    if mode != "fit_selector":
        optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    else:
        optimizer = Adam(params=model.selector.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1) * (sqrt(sqrt(sqrt(10))) ** min(x, 50)), 1)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()
    step = 0
    count = 0
    t0 = time()
    losssum = 0
    val_losssum = 0
    gradsum = 0
    loss_dict = None
    val_loss_dict = None
    modcount = 0
    simple_logger = SimpleLogger("loss_csv/{}.csv".format(modelname),
                                 ['step', 'loss', 'val_loss', 'grad_norm', 'learning_rate'])

    for i in tqdm(range(int(steps / BATCH_SIZE / SEQ_LEN))):

        step += 1
        spatial, nonspatial, prev_action, act, hidden = train_loader.get_batch(BATCH_SIZE)

        verb_print('batchsize ', BATCH_SIZE)
        verb_print('sequence length ', SEQ_LEN)
        verb_print('pov_shape: ', spatial.shape)
        verb_print('nonspatial_shape: ', nonspatial.shape)
        verb_print('act_shape: ', act.shape)

        count += BATCH_SIZE * SEQ_LEN
        modcount += BATCH_SIZE * SEQ_LEN

        loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden,
                                             torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

        loss_dict = update_loss_dict(loss_dict, ldict)
        train_loader.put_back(hidden)

        loss = loss.sum()  # / BATCH_SIZE / SEQ_LEN
        loss.backward()

        losssum += loss.item()

        # if mode == "fit_selector":
        #    grad_norm = clip_grad_norm_(model.selector.parameters(), 10)
        # else:
        grad_norm = clip_grad_norm_(model.parameters(), 10)

        gradsum += grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        ### Eval on one batch #####

        model.eval()

        # get batch
        spatial, nonspatial, prev_action, act, hidden = val_loader.get_batch(BATCH_SIZE)

        # forward
        val_loss, val_ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden,
                                                     torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)

        loss_dict = update_loss_dict(val_loss_dict, val_ldict)
        val_loader.put_back(hidden)

        val_loss = val_loss.sum()  # / BATCH_SIZE / SEQ_LEN

        val_losssum += val_loss.item()

        model.train()

        # end eval #######

        # print('count/(steps/20): ',count//int(steps/20))
        # print('count: ', count )
        # print('modcount: ', modcount)

        if modcount >= steps / 20:
            if ONLINE:
                print("------------------Saving Model!-----------------------")
                torch.save(model.state_dict(), "train/{}/{}.tm".format(modelname, modelname))
                torch.save(model.state_dict(),
                           "train/{}/{}_{}.tm".format(modelname, modelname, count // int(steps / 20)))

            modcount -= int(steps / 20)

            # What is this?!
            if ONLINE:
                if count // int(steps / 20) == number_of_checkpoints:
                    break

        if step % 40 == 0:
            # print(losssum, count, count/(time()-t0))
            if trains_loaded and not ONLINE:
                pass
            #   for k in loss_dict:
            #       logger.report_scalar(title='Training_'+mode, series='loss_'+k, value=loss_dict[k]/40, iteration=int(count))
            #   logger.report_scalar(title='Training_'+mode, series='loss', value=losssum/40, iteration=int(count))
            #   logger.report_scalar(title='Training_'+mode, series='grad_norm', value=gradsum/40, iteration=int(count))
            #   logger.report_scalar(title='Training_'+mode, series='learning_rate', value=float(optimizer.param_groups[0]["lr"]), iteration=int(count))

            print("-------------Logging!!!-------------")
            simple_logger.log(
                [step, losssum / 40, val_losssum / 40, gradsum / 40, float(optimizer.param_groups[0]["lr"])])
            losssum = 0
            val_losssum = 0
            gradsum = 0
            loss_dict = None
            val_loss_dict = None
        #  if mode == "fit_selector":
        #      torch.save(model.state_dict(),"train/model_fitted.tm")
        #  else:
        #      torch.save(model.state_dict(), "train/some_model.tm")


def main():
    # a bit of code that creates clearml logging (formerly trains) if clearml
    # is available
    if trains_loaded and not ONLINE:
        task = Task.init(project_name='MineRL', task_name='kmeans pic+pic+tre 1024 + flips whatever')
        logger = task.get_logger()
    else:
        logger = None

    os.makedirs("train", exist_ok=True)
    cached_kmeans("train", "MineRLObtainDiamondVectorObf-v0")
    print("lets gooo", file=sys.stderr)

    # train_files = absolute_file_paths('data/MineRLTreechopVectorObf-v0')
    train_files = absolute_file_paths('data/MineRLTreechop-v0/train')
    val_files = absolute_file_paths('data/MineRLTreechop-v0/val')
    model = Model(deviceStr=deviceStr, verbose=True, no_classes=30, with_masks=with_masks)

    shuffle(train_files)
    shuffle(val_files)

    train_loader = BatchSeqLoader(16, train_files, SEQ_LEN, model)
    val_loader = BatchSeqLoader(16, val_files, SEQ_LEN, model)
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
    train(model, "train", 150000000, train_loader, val_loader,
          logger)
    print('training done!')
    torch.save(model.state_dict(), "train/some_model.tm")
    print("ok", file=sys.stderr)

    train_loader.kill()
    val_loader.kill()


if __name__ == "__main__":
    main()
