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
import torch.distributions as D
from torch.nn import Sigmoid
import coloredlogs

coloredlogs.install(logging.WARNING)

from model import Model
import torch
import cv2
from record_episode import EpisodeRecorder
from descrete_actions_transform import transform_actions, transform_onehot_to_actions, transform_to_actions

import sys

import argparse
import ast

import simple_logger
import re
from tqdm import tqdm, trange
from simple_logger import SimpleLogger
from mine_env_creator import set_env_pos
from mineDataset import MineDataset

parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('modelpath', help='use saved model at path', type=str)
parser.add_argument('--test-epochs', help='test every x epochs', type=int, default=2)
parser.add_argument('--no-classes', help="how many actions are there?", type=int, default=30)
parser.add_argument('--no-cpu', help="make torch use this number of threads", type=int, default=4)
parser.add_argument('--verbose', help="print more stuff", action="store_true")
parser.add_argument('--with-masks', help="use extra mask channel", action="store_true")
parser.add_argument('--save-vids', help="save videos of eval", action="store_true")
parser.add_argument('--max-steps', help="max steps per episode", type=int, default=1000)
parser.add_argument('--sequence-len', help="reset states after how many steps?!", type=int, default=1000)
parser.add_argument('--num-threads', help="how many eval threads?", type=int, default=1)
parser.add_argument('--multilabel-prediction', help="model predicts multilabel and not discrete", action="store_true")
parser.add_argument('--manual-frameskip', help="use manual framskip for predictions", action="store_true")
parser.add_argument('--accumulate-prob', help="accumulate probabilities over number of actions", type = int, default=0)

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
multilabel = args.multilabel_prediction
manual_frameskip = args.manual_frameskip
accumulate_probs = args.accumulate_prob

torch.set_num_threads(no_cpu)

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')

print(MINERL_GYM_ENV)

device = "cuda" if torch.cuda.is_available() else "cpu"

# not tested seeds. Probably instant death
seeds = [303544, 744421, 816128, 406373, 99999, 88888, 76543, 927113, 873766, 11342, 82666,
         76543]

pos = [{'x': 50, 'y': 50, 'z': 50},
       {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50},
       {'x': 50, 'y': 50, 'z': 50},
       {'x': 50, 'y': 50, 'z': 50}, {'x': 50, 'y': 50, 'z': 50}]

# tested seeds.
seeds = [2, 2, 2]
pos = [{'x': -39, 'y': 59, 'z': 221}, {'x': -161, 'y': 60, 'z': 230}, {'x': -166, 'y': 63, 'z': 210}]


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

    seq_regex = re.compile(r'_seq-len=-?[0-9]*_')
    seq_len = seq_regex.search(name).group()
    seq_len = find_int(seq_len)

    masks_regex = re.compile(r'_with-masks=(False|True)_')
    with_masks = masks_regex.search(name).group()
    with_masks = (re.compile(r'(False|True)').search(with_masks).group())

    lstm_regex = re.compile(r'_with-lstm=(False|True)_')
    with_lstm = lstm_regex.search(name).group()
    with_lstm = re.compile(r'(False|True)').search(with_lstm).group()

    if with_masks == 'True':
        with_masks = True
    else:
        with_masks = False

    if with_lstm == 'True':
        with_lstm = True
    else:
        with_lstm = False

    modeldict = dict()
    modeldict['name'] = name
    modeldict['epoch'] = epoch
    modeldict['with_masks'] = with_masks
    modeldict['no_classes'] = no_classes
    modeldict['seq_len'] = seq_len
    modeldict['with_lstm'] = with_lstm

    return modeldict


def accumulate_actions(logit_list,no_actions):
    s = torch.zeros(no_actions)

    # add weighted logits
    for i,logits in enumerate(reversed(logit_list)):
        s+= Sigmoid()(logits.squeeze())*(1/(2**i))

    # mean
    s = s / len(logit_list)

    # sample
    s = D.bernoulli.Bernoulli(probs=s).sample().cpu().numpy()


    return s

def main():
    reward_logger_lock = Lock()
    act_logger_lock = Lock()
    mp4_lock = Lock()

    def thread_eval_on_env(models, env, seed):

        env.make_interactive(port=6666, realtime=True)

        # manual frameskip setup

        action_names = {0: 'attack', 1: 'back', 2: 'forward', 3: 'jump', 4: 'left', 5: 'right', 6: 'sneak', 7: 'sprint',
                        8: 'pitch_positive', 9: 'pitch_negative', 10: 'yaw_positive', 11: 'yaw_negative'}

#label means
#[[0.6728029  0.00472992 0.27164885 0.04502885 0.0303661  0.02564563
#  0.00161763 0.03583388 0.14151925 0.1690947  0.14615457 0.14349636]]


# [[0.43203875 0.01666584 0.4586578  0.0600334  0.06072781 0.05551973
#   0.01125936 0.05024552 0.19117768 0.25987467 0.19243424 0.21895409]]

        frameskips =np.array([30,1,1,1,1,1,1,1,1,1,1,1])


        er = EpisodeRecorder()

        for modeldict in models:
            print('loop')

            modelname = modeldict['name']
            epoch = modeldict['epoch']
            print(f"loading model {modeldict['name']}")

            print(f"testing on seed {seed} ")
            model = Model(deviceStr=device, verbose=verbose, no_classes=modeldict['no_classes'],
                          with_masks=modeldict['with_masks'], mode='eval', with_lstm=modeldict['with_lstm'])
            model.load_state_dict(torch.load(os.path.join(modelpath, modeldict['name']), map_location=device),
                                  strict=False)

            if torch.equal(model.logits_mean, torch.zeros(modeldict['no_classes'])):
                compute_logits_mean(model)
                torch.save_to_state_dict(model.state_dict(), os.path.join(modelpath, modeldict['name']))
                print(f"mean_computed")
                continue

            print(model.logits_mean)
            model.eval()

            print(f"loaded model {modeldict['name']}")

            logit_list= []

            with torch.no_grad():
                print(f"starting eval on {modeldict['name']} , epoch = {modeldict['epoch']}, seed= {seed}")
                env.seed(seed)
                obs = env.reset()

                state = model.get_zero_state(1, device)

                rewards = []


                ## manual frameskips

                if manual_frameskip:
                    framecount_per_action = np.zeros(12)
                    current_action = np.zeros(12)


                for step in range(max_steps):

                    pov = torch.tensor(obs['pov'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

                    if not multilabel:
                        s, p, state = model.sample(pov, additional_info_dummy, state, mean_substract=True)
                        action = transform_to_actions([int(s)], no_actions=modeldict['no_classes'])

                    else:


                        s, p, state, logits = model.sample_multilabel(pov, additional_info_dummy, state, mean_substract=False)

                        logit_list.append(logits)

                        if accumulate_probs:
                            s = accumulate_actions(logit_list[step-accumulate_probs:],no_classes)

                        if manual_frameskip:
                            # check if framskip over for all actions. Use long and not bool for more computation possibilities
                            mask = (framecount_per_action >= frameskips)

                            # reset skipped actionsno
                            current_action = np.logical_and(current_action,~mask).astype(np.long)

                            # get rid of -1's.
                            current_action [current_action < 0] = 0

                            # update current action with sampled action
                            current_action = np.logical_or(current_action,s).astype(np.long)


                            print(f'framecount:{framecount_per_action}')
                            print(f'sampled:{s}')
                            print(f'current:{current_action}')

                            # reset framseskips that reached threshold
                            framecount_per_action[mask] = 0

                            # add executed aciton to count
                            framecount_per_action += current_action

                            s = current_action

                        action = transform_to_actions(s, no_actions=modeldict['no_classes'], use_vecs=True)

                    # print(int(s))
                    obs, reward, done, _ = env.step(action)

                    rewards.append(reward)
                    er.record_frame(obs['pov'])

                    act_logger_lock.acquire()
                    if multilabel:
                        action_logger.log([modeldict['name'],manual_frameskip,accumulate_probs, modeldict['epoch'], seed, p, s, step, reward])
                    else:
                        action_logger.log([modeldict['name'],manual_frameskip,accumulate_probs, modeldict['epoch'], seed, int(p), int(s), step, reward])
                    act_logger_lock.release()

                    if done:
                        break

                    if step % modeldict['seq_len'] == 0:
                        state = model.get_zero_state(1, device)

                mp4_lock.acquire()

                er.save_vid(f'eval/{modelname}/epoch={epoch}_seed={seed}_manual-frameskip={manual_frameskip}_accumulate_probs={accumulate_probs}.mp4')
                mp4_lock.release()

                er.reset()

                reward_logger_lock.acquire()
                rewards_logger.log(
                    [modeldict['name'], modeldict['epoch'],manual_frameskip,accumulate_probs, seed, np.mean(rewards), np.std(rewards), np.var(rewards),
                     np.sum(rewards)])
                reward_logger_lock.release()

    rewards_logger = SimpleLogger(f'eval/{model_name}/rewards.csv',
                                  ['modelname','manual_frameskip','accumulate_probs', 'epoch', 'seed', 'reward_mean', 'reward_sdt', 'reward_var',
                                   'reward_sum'])
    action_logger = SimpleLogger(f'eval/{model_name}/actions.csv',
                                 ['modelname','manual_frameskip','accumulate_probs','epoch', 'seed', 'predicted_action', 'sampled_action', 'step', 'reward'])

    additional_info_dummy = torch.zeros(10)

    # dicts for thread!
    models = []

    ## find all parameters

    for modelname_epoch in os.listdir(modelpath):

        if modelname_epoch.split('/')[-1].split('.')[-1] != 'tm':
            continue

        modeldict = get_model_info_from_name(modelname_epoch)

        if modeldict['epoch'] not in [6,7,8,9,10]:
            continue

        models.append(modeldict)

    threads = []
    envs = []
    for seed, posit in zip(seeds, pos):

        #    set_env_pos(posit)
        set_env_pos(posit)

        env = gym.make(MINERL_GYM_ENV)
        env.seed(seed)
        envs.append(env)

        p = Process(target=thread_eval_on_env, args=(models, env, seed,))
        threads.append(p)
        print('starting thread!')
        p.start()

        if len(threads) >= num_threads:
            print(f'waiting for {len(threads)} threads')
            for thread in threads:
                thread.join()
            for i in range(len(envs)):
                envs[i].close()
                del envs[i]

            envs = []
            threads = []

    for thread in threads:
        thread.join()
        pass


def compute_logits_mean(model):
    ds = MineDataset('data/MineRLTreechop-v0/train', no_replays=3, random_sequences=None, sequence_length=-1,
                     device='cpu', with_masks=False, min_reward=1, min_variance=20, ros=False)

    logits_list = []

    for pov, _ in ds:
        pov = pov.unsqueeze(0).unsqueeze(0)
        pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()
        logits, _ = model.forward(pov, torch.zeros(10), model.get_zero_state(1, device='cpu'))
        logits_list.append(logits)

    logits = torch.cat(logits, dim=0)
    logits = torch.mean(logits, dim=0)

    print(logits_list)


if __name__ == "__main__":
    main()
