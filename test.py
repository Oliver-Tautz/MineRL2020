
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


parser = argparse.ArgumentParser(description='train the model ...')
parser.add_argument('modelname',help="name of the model",type=str)
parser.add_argument('--no_episodes',help="how many episodes to test per model?",type=int,default=10)
parser.add_argument('--no-classes',help="how many actions are there?",type=int,default=30)
parser.add_argument('--no-cpu',help="number of eval threads to start",type=int,default=2)
parser.add_argument('--verbose',help="print more stuff",action="store_true")
parser.add_argument('--test-checkpoints',help="test checkpoints?",action="store_true")
parser.add_argument('--checkpoint-range',help="Range in format \"[i,j,k,...]\"",type=str,default='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]')
parser.add_argument('--with-masks',help="use extra mask channel",action="store_true")
parser.add_argument('--save-vids',help="save videos of eval",action="store_true")
parser.add_argument('--max-steps',help="max steps per episode",type=int,default=3000)


args = parser.parse_args()



if args.test_checkpoints:


    checkpoint_ix = ast.literal_eval(args.checkpoint_range)

else:
    checkpoint_ix = [0]



model_name = args.modelname
test_checkpoints = args.test_checkpoints


sl = simple_logger.SimpleLogger(f"eval_csv/{model_name}.csv",['checkpoint','score','episodes','steps'])





# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_MAX_EVALUATION_EPISODES = args.no_episodes#int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 10))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = args.no_cpu#int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 4))

device = "cuda" if torch.cuda.is_available() else "cpu"


class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.
    
    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class 

    This class enables the evaluator to run your agent in parallel, 
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs)) 
                ...
        
        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################

class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent. 
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64*64*3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size))*2 -1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten()/255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1,1)}


    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs,reward,done,_ = single_episode_env.step(self.act(self.flatten_obs(obs)))

rewards = []

class MineRLNetworkAgent(MineRLAgentBase):
    """
    An example random agent. 
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self,model_name,checkpoint_number):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        self.model = Model(deviceStr=device,verbose=args.verbose,no_classes=args.no_classes,with_masks=args.with_masks)
        self.checkpoint_number = checkpoint_number

        if test_checkpoints:
            self.model.load_state_dict(torch.load(f"train/{model_name}/{model_name}_{checkpoint_number}.tm", map_location=device))

        else:
            self.model.load_state_dict(torch.load("train/unimportant/unimportant_with-masks]False_map-to-zero=False_no-classes=50_epoch=1_time=2021-06-08 15:40:30.410176.tm", map_location=device))
        # self.model.load_state_dict(torch.load("testing/m.tm", map_location=device))

        self.model.eval()
        self.model.to(device)


    def run_agent_on_episode(self, single_episode_env : Episode,index):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        reward_sum = 0
        counter = 0
        er = EpisodeRecorder()


        max_steps = args.max_steps
        seeds = [2, 12345, 45678, 303544, 744421, 816128, 406373, 99999, 88888, 76543,
                 67345, 11342, 24337, 45547, 82666, 128939, 494668, 927113, 869989, 873766, 708928, 786039, 371309,
                 281286]

        with torch.no_grad():
            seed = random.sample(seeds,1)
            single_episode_env.seed(seed)
            obs = single_episode_env.reset()
            er.record_frame(obs['pov'])
            done = False
            state = self.model.get_zero_state(1, device=device)
            s = torch.zeros((1,1,64), dtype=torch.float32, device=device)
            while not done:

                # what is happening here?
                spatial = torch.tensor(obs["pov"], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).transpose(2,4)


                #cv2.imshow("xdd", obs["pov"])
                # cv2.waitKey(30)
                #nonspatial = torch.cat([torch.tensor(obs["vector"], device=device, dtype=torch.float32),
                 #                       torch.ones((2,), device=device,dtype=torch.float32)], dim=0).unsqueeze(0).unsqueeze(0)


                nonspatial = torch.zeros((1,1,64))

                s, state = self.model.sample(spatial, nonspatial, s, state, torch.zeros((1,1,64),dtype=torch.float32,device=device))


                action = transform_int_to_actions([int(s)])


                for i in range(1):
                    obs,reward,done,_ = single_episode_env.step(action)
                    er.record_frame(obs['pov'])

                    counter+=1

                    reward_sum += reward
                    if reward > 0:
                        rewards.append(reward)
                    if done :
                        break
                    if max_steps and counter > max_steps:
                        done = True

        if args.save_vids:
            er.save_vid(f'train/{model_name}/eval_vids/checkpoint={self.checkpoint_number}/episode={index}_seed={seed}.avi')
            sl = simple_logger.SimpleLogger(f"eval_csv/{model_name}.csv", ['checkpoint', 'score', 'episodes', 'steps'])
        sl.log([self.checkpoint_number,reward_sum,1,max_steps])
        
        

class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)
        
#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   # 
######################################################################
AGENT_TO_TEST = MineRLNetworkAgent # MineRLMatrixAgent, MineRLRandomAgent, YourAgentHere



####################
# EVALUATION CODE  #
####################
def main():

    for check_no in checkpoint_ix:
        agent = AGENT_TO_TEST()
        assert isinstance(agent, MineRLAgentBase)
        agent.load_agent(model_name,check_no)

        assert MINERL_MAX_EVALUATION_EPISODES > 0
        assert EVALUATION_THREAD_COUNT > 0

        # Create the parallel envs (sequentially to prevent issues!)
        envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]




        episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
        episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
        # A simple funciton to evaluate on episodes!

        def evaluate(i, env):
            print("[{}] Starting evaluator.".format(i))

            for i in range(episodes_per_thread[i]):
                try:
                    agent.run_agent_on_episode(Episode(env),i)
                except EpisodeDone:
                    print("[{}] Episode complete".format(i))

                    pass

        evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
        for thread in evaluator_threads:
            thread.start()

        # wait fo the evaluation to finish
        for thread in evaluator_threads:
            thread.join()

        print("average:", sum(rewards)/MINERL_MAX_EVALUATION_EPISODES)

if __name__ == "__main__":
    main()
    

