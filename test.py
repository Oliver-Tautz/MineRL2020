import json
import select
import time
import logging
import os

os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Process, Lock

from typing import Callable

import gym
import minerl
import abc
import numpy as np
import random

import coloredlogs

coloredlogs.install(logging.DEBUG)

from model import Model
import torch
import cv2
from record_episode import EpisodeRecorder
from descrete_actions_transform import transform_actions, transform_onehot_to_actions, transform_int_to_actions

import sys

import argparse
import ast

import simple_logger
import re
from tqdm import tqdm, trange
from simple_logger import SimpleLogger
from mine_env_creator import set_env_pos

parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('modelpath', help='use saved model at path', type=str)
parser.add_argument('--test-epochs', help='test every x epochs', type=int, default=4)
parser.add_argument('--no-classes', help="how many actions are there?", type=int, default=30)
parser.add_argument('--no-cpu', help="make torch use this number of threads", type=int, default=4)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--save-vids', help="save videos of eval", action="store_true")
parser.add_argument('--max-steps', help="max steps per episode", type=int, default=1000)
parser.add_argument('--sequence-len', help="reset states after how many steps?!", type=int, default=1000)
parser.add_argument('--num-threads', help="how many eval threads?", type=int, default=1)

args = parser.parse_args()

modelpath = args.modelpath
model_name = modelpath.split('/')[-1].split('.')[0]
no_classes = args.no_classes
no_cpu = args.no_cpu
verbose = args.verbose
with_masks = args.with_masks
save_vids = args.save_vids
max_steps = args.max_steps
test_epochs = args.test_epochs
num_threads = args.num_threads

torch.set_num_threads(no_cpu)

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')

print(MINERL_GYM_ENV)

device = "cuda" if torch.cuda.is_available() else "cpu"


# not tested seeds. Probably instant death
seeds = [ 303544, 744421, 816128, 406373, 99999, 88888, 76543, 927113, 873766, 11342, 82666,
         76543]

pos =  [ {'x': 50, 'y': 50, 'z': 50},
       {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50},
       {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50}]

# tested seeds.
seeds = [2,12345, 45678]
pos = [{'x': 100, 'y': 100, 'z': 100},{'x': 60, 'y': 60, 'z': 60},{'x': 1700, 'y': 80, 'z': 1060}]

def get_model_info_from_name(name):
    if name.split('/')[-1].split('.')[-1] != 'tm':
        return -1

    # THIS PARSING IS NOT ROBUST AT ALL AND ONLY WORKS WITH MY NAMING SHEME(HOPEFULLY).!
    def find_int(str):

        int_regex = re.compile(r'[0-9][0-9]*')
        return int(int_regex.search(str).group())

    ## check if regex in name!
    epoch_regex = re.compile(r'_epoch=[0-9]*_')

    epoch = epoch_regex.search(name).group()
    epoch = find_int(epoch)

    classes_regex = re.compile(r'_no-classes=[0-9]*_')
    no_classes = classes_regex.search(name).group()
    no_classes = find_int(no_classes)

    seq_regex = re.compile(r'_seq-len=[0-9]*_')
    seq_len = seq_regex.search(name).group()
    seq_len = find_int(seq_len)

    masks_regex = re.compile(r'_with-masks=(False|True)_')
    with_masks = masks_regex.search(name).group()
    with_masks = (re.compile(r'(False|True)').search(with_masks).group())

    if with_masks == 'True':
        with_masks = True
    else:
        with_masks = False

    modeldict = dict()
    modeldict['name'] = name
    modeldict['epoch'] = epoch
    modeldict['with_masks'] = with_masks
    modeldict['no_classes'] = no_classes
    modeldict['seq_len'] = seq_len

    return modeldict


def main():
    reward_logger_lock = Lock()
    act_logger_lock = Lock()
    mp4_lock = Lock()

    def thread_eval_on_env(models, env, seed):

        er = EpisodeRecorder()

        for modeldict in models:
            print('loop')

            modelname = modeldict['name']
            epoch = modeldict['epoch']
            print(f"loading model {modeldict['name']}")

            print(f"testing on seed {seed} ")
            model = Model(deviceStr=device, verbose=verbose, no_classes=modeldict['no_classes'],
                          with_masks=modeldict['with_masks'], mode='eval')
            model.load_state_dict(torch.load(os.path.join(modelpath, modeldict['name']), map_location=device))
            model.eval()

            print(f"loaded model {modeldict['name']}")

            with torch.no_grad():
                print(f"starting eval on {modeldict['name']} , epoch = {modeldict['epoch']}, seed= {seed}")
                #env.seed(seed)
                obs = env.reset()

                state = model.get_zero_state(1, device)

                rewards = []
                for step in range(max_steps):

                    pov = torch.tensor(obs['pov'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()
                    s, p, state = model.sample(pov, additional_info_dummy, state)
                    action = transform_int_to_actions([int(s)], no_actions=modeldict['no_classes'])

                    obs, reward, done, _ = env.step(action)

                    rewards.append(reward)
                    er.record_frame(obs['pov'])

                    act_logger_lock.acquire()
                    action_logger.log([modeldict['name'], modeldict['epoch'], seed, int(p), int(s), step, reward])
                    act_logger_lock.release()

                    if done:
                        break

                    if step % modeldict['seq_len'] == 0:
                        state = model.get_zero_state(1, device)

                mp4_lock.acquire()

                er.save_vid(f'eval/{modelname}/epoch={epoch}_seed={seed}.mp4')
                mp4_lock.release()

                er.reset()

                reward_logger_lock.acquire()
                rewards_logger.log(
                    [modeldict['name'], modeldict['epoch'], seed, np.mean(rewards), np.std(rewards), np.var(rewards),
                     np.sum(rewards)])
                reward_logger_lock.release()


    rewards_logger = SimpleLogger(f'eval/{model_name}/rewards.csv',
                                  ['modelname', 'epoch', 'seed', 'reward_mean', 'reward_sdt', 'reward_var',
                                   'reward_sum'])
    action_logger = SimpleLogger(f'eval/{model_name}/actions.csv',
                                 ['modelname', 'epoch', 'seed', 'predicted_action', 'sampled_action', 'step', 'reward'])

    additional_info_dummy = torch.zeros(10)

    # dicts for thread!
    models = []

    ## find all parameters

    for modelname_epoch in os.listdir(modelpath):

        if modelname_epoch.split('/')[-1].split('.')[-1] != 'tm':
            continue

        modeldict = get_model_info_from_name(modelname_epoch)

        if modeldict['epoch'] % test_epochs != 0:
            continue

        models.append(modeldict)

    threads = []
    envs = []
    for seed, posit in zip(seeds, pos):

        set_env_pos(posit)
        env = gym.make(MINERL_GYM_ENV)
        env.seed(seed)
        envs.append(env)

        p = Process(target=thread_eval_on_env, args=(models[0:1],env,seed,))
        threads.append(p)
        print('starting thread!')
        p.start()

        if len(threads) >= num_threads:
            print(f'waiting for {len(threads)} threads')
            for thread in threads:
                thread.join()
            for i in range(len(envs)):
                del envs[i]

            envs  = []
            threads = []

    for thread in threads:
        thread.join()
        pass


if __name__ == "__main__":
    main()
