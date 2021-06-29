# simple wrapper function for mineRL environment to set different starting positions
import os.path

import gym
from lxml import etree as ET
import shutil
from os.path import isfile


# simple method to set seed and pos in env. WARNING! Overrides global .xml by default. Saves old xml's in baks!
def create_env(env_name, seed, pos,
               xml_path='/home/olli/gits/Minecraft72/venv/lib/python3.9/site-packages/minerl/herobraine/env_specs/missions/treechop.xml'):


    # manage baks. Make new bak file if already present!
    counter = 0
    while os.path.isfile(f"{xml_path}.bak_{counter}"):
        counter+=1


    shutil.copy(xml_path,f"{xml_path}.bak_{counter}")

    # change/add placement in .xml file

    tree = ET.parse(xml_path)
    root = tree.getroot()
    agent_section =     root.find('{http://ProjectMalmo.microsoft.com}AgentSection')
    agent_start = agent_section.find('{http://ProjectMalmo.microsoft.com}AgentStart')

    placement = agent_start.find('{http://ProjectMalmo.microsoft.com}Placement')


    if  placement is None:
        placement = ET.Element('{http://ProjectMalmo.microsoft.com}Placement')
        agent_start.append(placement)

    placement.set('x', str(pos['x']))
    placement.set('y', str(pos['y']))
    placement.set('z', str(pos['z']))

    placement.set('yaw', '0')
    placement.set('pitch', '0')

    ###

    # write changes to file
    tree.write(xml_path,pretty_print=True)

    # make env
    env = gym.make(env_name)

    # set seed
    env.seed(seed)

    # reset. Is this required?
    env.reset()

    return env
