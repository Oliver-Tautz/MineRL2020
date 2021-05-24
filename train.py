# Simple env test.
import json
import select
import time
import logging
import os
import sys

from tqdm import tqdm
import cProfile as profile

import gym
import minerl
import coloredlogs

from torchsummary import summary

coloredlogs.install(logging.DEBUG)

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


# In ONLINE=True mode the code saves only the final version with early stopping,
# in ONLINE=False it saves 20 intermediate versions during training.
ONLINE = True

trains_loaded = True
verb=False

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
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
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


def train(model, mode, steps, loader, logger):

    ## Profiler
    pr = profile.Profile()
    pr.disable()

    torch.set_num_threads(1)
    if mode != "fit_selector":
        optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    else:
        optimizer = Adam(params=model.selector.parameters(), lr=1e-4, weight_decay=1e-6)

    def lambda1(x):
        return min((1e-1)* (sqrt(sqrt(sqrt(10)))**min(x, 50)),1)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    optimizer.zero_grad()
    step = 0
    count = 0
    t0 = time()
    losssum = 0
    gradsum = 0
    loss_dict = None
    modcount = 0
    simple_logger = SimpleLogger("loss_csv/{}.csv".format(sys.argv[1]),['step','loss','grad_norm','learning_rate'])



    for i in tqdm(range(int(steps/ BATCH_SIZE / SEQ_LEN))):


        verb_print('batchsize ',BATCH_SIZE)
        verb_print('sequence length ', SEQ_LEN)
        step+=1
        verb_print(i)
        spatial, nonspatial, prev_action, act,  hidden = loader.get_batch(BATCH_SIZE)

        verb_print('pov_shape: ',spatial.shape)
        verb_print('nonspatial_shape: ',nonspatial.shape)
        verb_print('act_shape: ',act.shape)


        count += BATCH_SIZE*SEQ_LEN
        modcount += BATCH_SIZE*SEQ_LEN

        pr.enable()

        if mode != "pretrain":
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, torch.zeros(act.shape, dtype=torch.float32, device=deviceStr), act)
        else:
            loss, ldict, hidden = model.get_loss(spatial, nonspatial, prev_action, hidden, act, act)

        pr.disable()
        pr.dump_stats('profile.pstat')
        loss_dict = update_loss_dict(loss_dict, ldict)
        loader.put_back(hidden)

        loss = loss.sum() # / BATCH_SIZE / SEQ_LEN
        loss.backward()
        
        losssum += loss.item()
        
        if mode == "fit_selector":
            grad_norm = clip_grad_norm_(model.selector.parameters(),10)
        else:
            grad_norm = clip_grad_norm_(model.parameters(),10)
        
        gradsum += grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


        #print('count/(steps/20): ',count//int(steps/20))
        #print('count: ', count )
        #print('modcount: ', modcount)


        if modcount >= steps/20:
            if ONLINE:

                print("------------------Saving Model!-----------------------")
                torch.save(model.state_dict(), "train/some_model.tm")
                torch.save(model.state_dict(),"testing/model_{}.tm".format(count//int(steps/20)))

            modcount -= int(steps/20)

            # What is this?!
            if ONLINE:
                if count//int(steps/20) == 14:
                    break

        if step % 40 == 0:
            #print(losssum, count, count/(time()-t0))
            if trains_loaded and not ONLINE:
                pass
             #   for k in loss_dict:
             #       logger.report_scalar(title='Training_'+mode, series='loss_'+k, value=loss_dict[k]/40, iteration=int(count))
             #   logger.report_scalar(title='Training_'+mode, series='loss', value=losssum/40, iteration=int(count))
             #   logger.report_scalar(title='Training_'+mode, series='grad_norm', value=gradsum/40, iteration=int(count))
             #   logger.report_scalar(title='Training_'+mode, series='learning_rate', value=float(optimizer.param_groups[0]["lr"]), iteration=int(count))

            print("-------------Logging!!!-------------")
            simple_logger.log([step,losssum/40,gradsum/40,float(optimizer.param_groups[0]["lr"])])
            losssum = 0
            gradsum = 0
            loss_dict = None
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
    cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
    print("lets gooo", file=sys.stderr)


    #train_files = absolute_file_paths('data/MineRLTreechopVectorObf-v0')
    train_files = absolute_file_paths('data/MineRLTreechop-v0')
    model = Model(deviceStr=deviceStr)


    shuffle(train_files)






    loader = BatchSeqLoader(16, train_files, SEQ_LEN, model)
    spatial, nonspatial, prev_action, act, hidden = loader.get_batch(BATCH_SIZE)

    #print(spatial.shape)
    #print(nonspatial.shape)
    #print(prev_action.shape)
    #print(act.shape)
    #print(hidden.shape)

    #summary(Model)

    #if LOAD:
    #    model.load_state_dict(torch.load("train/some_model.tm"))
    if deviceStr == "cuda":
        model.cuda()
    else:
        model.cpu()

    print('Starting training!')
    train(model, "train", 150000000, loader, logger)
    print('training done!')
    torch.save(model.state_dict(), "train/some_model.tm")
    print("ok", file=sys.stderr)

    loader.kill()


if __name__ == "__main__":
    main()
