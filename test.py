
import json
import select
import time
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
import threading


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
from tqdm import tqdm
from simple_logger import SimpleLogger

parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('modelpath',help='use saved model at path',type=str)
parser.add_argument('--test-epochs',help='test every x epochs',type=int,default=5)
parser.add_argument('--no-classes',help="how many actions are there?",type=int,default=30)
parser.add_argument('--no-cpu',help="number of eval threads to start. Doesnt do anything yet!",type=int,default=1)
parser.add_argument('--verbose',help="print more stuff",action="store_true")
parser.add_argument('--with-masks',help="use extra mask channel",action="store_true")
parser.add_argument('--save-vids',help="save videos of eval",action="store_true")
parser.add_argument('--max-steps',help="max steps per episode",type=int,default=10000)
parser.add_argument('--sequence-len',help="reset states after how many steps?!",type=int,default=100)

args = parser.parse_args()



modelpath = args.modelpath
model_name = modelpath.split('/')[-1].split('.')[0]
no_classes = args.no_classes
no_cpu = args.no_cpu
verbose = args.verbose
with_masks=args.with_masks
save_vids = args.save_vids
max_steps=args.max_steps
test_epochs = args.test_epochs










# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')




device = "cuda" if torch.cuda.is_available() else "cpu"





####################
# EVALUATION CODE  #
####################
def main():

        rewards_logger = SimpleLogger(f'eval/{model_name}/rewards.csv',
                               ['modelname','epoch','seed','reward_sum','reward_sdt','reward_var','reward_sum'])
        action_logger = SimpleLogger(f'eval/{model_name}/actions.csv',
                               ['modelname','epoch','seed','predicted_action','sampled_action','step','reward'])
        er = EpisodeRecorder()
        seeds = [2, 12345, 45678, 303544, 744421, 816128, 406373, 99999, 88888, 76543, 927113, 873766, 11342, 82666,
                 76543]

        additional_info_dummy = torch.zeros(10)
        env = gym.make(MINERL_GYM_ENV)


        for modelname_epoch in os.listdir(modelpath):


            ## check of regex in name!
            epoch_regex = re.compile(r'_epoch=[0-9]*_')
            int_regex = re.compile(r'[0-9][0-9]*')
            epoch = epoch_regex.search(modelname_epoch).group()
            epoch = int(int_regex.search(epoch).group())

            if epoch % test_epochs !=0:
                continue

            model = Model(deviceStr=device, verbose=verbose, no_classes=no_classes, with_masks=with_masks,mode='eval')
            model.load_state_dict(torch.load(os.path.join(modelpath,modelname_epoch), map_location=device))
            model.eval()


            with torch.no_grad():
                for seed in tqdm(seeds,desc='testing seeds'):
                    env.seed(seed)
                    obs= env.reset()

                    state = model.get_zero_state(1,device)

                    rewards = []
                    for step in range(max_steps):

                        pov = torch.tensor(obs['pov'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        pov = pov.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()
                        s,p, state = model.sample(pov,additional_info_dummy,state)
                        action = transform_int_to_actions([int(s)],no_actions=no_classes)

                        obs, reward, done, _ = env.step(action)

                        rewards.append(reward)
                        er.record_frame(obs['pov'])
                        action_logger.log([model_name,epoch,seed,int(p),int(s),step,reward])
                        if done:
                            break

                        if step % 99 ==0:
                            state  = model.get_zero_state(1,device)

                    er.save_vid(f'eval/{model_name}/{seed}.mp4')
                    er.reset()

                    rewards_logger.log([model_name,epoch,seed,np.mean(rewards),np.std(rewards),np.var(rewards),np.sum(rewards)])


if __name__ == "__main__":
    main()
    

