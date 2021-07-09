import minerl
import gym
from descrete_actions_transform import transform_int_to_actions
from mine_env_creator import set_env_pos
import numpy     as np
import logging
import coloredlogs
import time

coloredlogs.install(logging.DEBUG)

set_env_pos(None)
env = gym.make('MineRLTreechop-v0')



while True:
    obs = env.reset()
    done = False
    net_reward = 0
    while not done:

        action = transform_int_to_actions([np.random.randint(0,30)])
       # time.sleep(2)


        obs, reward, done, info = env.step(
            action)


        net_reward += reward
