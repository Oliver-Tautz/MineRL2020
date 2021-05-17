import os
import minerl
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def print_portions(camera_actions):
    pitch = camera_actions[:,0]
    yaw   = camera_actions[:,1]
    no_actions = yaw.shape[0]

    smallest_pitch = pitch[np.absolute(pitch)<0.1].shape[0]/no_actions
    smallest_yaw = yaw[np.absolute(yaw)<0.1].shape[0]/no_actions

    print('smallest <0.1:',  smallest_pitch,smallest_yaw)

    small_pitch = pitch[np.absolute(pitch)<5].shape[0]/no_actions
    small_yaw = yaw[np.absolute(yaw)<5].shape[0]/no_actions

    print('small<5:',  small_pitch,small_yaw)

    big_pitch = pitch[np.absolute(pitch)>5].shape[0]/no_actions
    big_yaw = yaw[np.absolute(yaw)>5].shape[0]/no_actions

    print('big<5:',  big_pitch,big_yaw)

    biggest_pitch = pitch[np.absolute(pitch)>20].shape[0]/no_actions
    biggest_yaw = yaw[np.absolute(yaw)>20].shape[0]/no_actions

    print('biggest>20:',  biggest_pitch,biggest_yaw)

    giant_pitch = pitch[np.absolute(pitch)>50].shape[0]/no_actions
    giant_yaw = yaw[np.absolute(yaw)>50].shape[0]/no_actions

    print('giant>50:',  giant_pitch,giant_yaw)

def plot_hists(camera_actions) :
    
    plt.hist(camera_actions[:,0],'sqrt')
    plt.savefig('dataset_plots/camera_pitch_hist.pdf')
    plt.clf()

    plt.hist(camera_actions[:,1],'sqrt')
    plt.savefig('dataset_plots/camera_yaw_hist.pdf')
    plt.clf()

    plt.hist(camera_actions[:,0],'sqrt',range=(-20,20))
    plt.savefig('dataset_plots/camera_pitch_hist_small.pdf')
    plt.clf()

    plt.hist(camera_actions[:,1],'sqrt',range=(-20,20))
    plt.savefig('dataset_plots/camera_yaw_hist_small.pdf')
    plt.clf()

    plt.hist(camera_actions[:,0],'sqrt',range=(-0.25,0.25))
    plt.savefig('dataset_plots/camera_pitch_hist_smaller.pdf')
    plt.clf()

    plt.hist(camera_actions[:,1],'sqrt',range=(-0.25,0.25))
    plt.savefig('dataset_plots/camera_yaw_hist_smaller.pdf')
    plt.clf()

def concat_actions(actlist):
    if actlist:
        start = actlist.pop()
    else:
        print('No actions given!')
        return []

    while actlist:
        start = np.concatenate((start,actlist.pop()),axis=0)

    return start


loader = minerl.data.make('MineRLTreechop-v0',data_dir='./data',num_workers=1)




no_streams = 10
actions = defaultdict(lambda : [])

# concats numpy arrays given in list.
# works


def map_actions(actions):
    pass

def discreticize_camera_action(camera_actions):

    pitch = camera_actions[:,0]
    yaw   = camera_actions[:,1]

    pitch[np.absolute(pitch) < 5 ] = 0
    pitch[pitch < -20]             = 1
    pitch[pitch < -50]             = 2
    pitch[pitch > 20]              = 3
    pitch[pitch > 50]              = 4
    
    yaw[np.absolute(yaw) < 5 ] = 0
    yaw[yaw < -20]             = 1
    yaw[yaw < -50]             = 2
    yaw[yaw > 20]              = 3
    yaw[yaw > 50]              = 4
    
    


for f in (os.listdir('./data/MineRLTreechop-v0')):
    d = loader._load_data_pyfunc('./data/MineRLTreechop-v0/{}'.format(f),-1,None)
    obs, act, reward, nextobs, done = d

    print(obs.keys())


    for key in act.keys():
        actions[key].append(act[key])

    no_streams-=1

    if no_streams <1 :
        break
        pass
# concat actions
for key in actions.keys():
        actions[key] = concat_actions(actions[key])
        print(actions[key].shape)



camera_actions = actions['camera']
print_portions(actions['camera'])
plot_hists((actions['camera']))

