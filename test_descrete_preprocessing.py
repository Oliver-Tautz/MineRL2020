import os
import minerl
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def print_portions(camera_actions):
    pitch = camera_actions[:,0]
    yaw   = camera_actions[:,1]
    no_actions = yaw.shape[0]

    smallest_pitch = pitch[np.absolute(pitch)<0.2].shape[0]/no_actions
    smallest_yaw = yaw[np.absolute(yaw)<0.2].shape[0]/no_actions

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

def plot_hists(camera_actions,without_noise=False) :
    pitch = camera_actions[:,0]
    yaw = camera_actions[:, 1]

    if without_noise:
        pitch=pitch[pitch>1]
        yaw=yaw[yaw>1]

    plt.hist(pitch,'sqrt')
    plt.savefig('dataset_plots/camera_pitch_hist.pdf')
    plt.clf()

    plt.hist(yaw,'sqrt')
    plt.savefig('dataset_plots/camera_yaw_hist.pdf')
    plt.clf()

    plt.hist(pitch,'sqrt',range=(-20,20))
    plt.savefig('dataset_plots/camera_pitch_hist_small.pdf')
    plt.clf()

    plt.hist(yaw,'sqrt',range=(-20,20))
    plt.savefig('dataset_plots/camera_yaw_hist_small.pdf')
    plt.clf()

    plt.hist(pitch,'sqrt',range=(-0.25,0.25))
    plt.savefig('dataset_plots/camera_pitch_hist_smaller.pdf')
    plt.clf()

    plt.hist(yaw,'sqrt',range=(-0.25,0.25))
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



# 210 max!
no_streams = 300
actions = defaultdict(lambda : [])



for f in tqdm(os.listdir('./data/MineRLTreechop-v0'),desc='loading'):
    d = loader._load_data_pyfunc('./data/MineRLTreechop-v0/{}'.format(f),-1,None)
    obs, act, reward, nextobs, done = d


    for key in act.keys():
        actions[key].append(act[key])

    no_streams-=1

    if no_streams <1 :
        break
        pass
# concat actions
print('action_keys: ',list(actions.keys()))

for key in actions.keys():
        actions[key] = concat_actions(actions[key])




camera_actions = actions['camera']
print_portions(actions['camera'])
#plot_hists((actions['camera']))

#pitch, yaw = discreticize_camera_action(actions['camera'], 0.2)

#print('pitch_shape: ',pitch.shape,'\nyaw_shape:', yaw.shape)

from descrete_actions_transform import transform_actions, save_frequent_actions_and_mapping,transform_onehot_to_actions




def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# map to zero performs better.
int_to_vec, key_to_ix = save_frequent_actions_and_mapping(actions)

def test(map_to_zero):
    X = transform_actions(actions, map_to_zero=map_to_zero)


    print(X.shape)
    #print(int_to_vec)
    #print(key_to_ix)
    reconstructed_actions = transform_onehot_to_actions(X)

    scores = []
    # test regular actions. looks good!
    for key in actions.keys():
        if key == 'camera':
            continue
        print("-------------------{}--------------".format(key))
        ac  = actions[key]
        rac = reconstructed_actions[key]

        correct = np.sum(np.logical_not(np.logical_xor(ac,rac)),axis=0)
        score= correct/len(ac)
        print("{}% correctly restored.".format(score))
        scores.append(score)



    def test_one_direction(c,rc):
        if np.absolute(c)>0.5:
            if c > 0 and rc > 0:
                return True
            elif c < 0and rc< 0:
                return 1
            else: return 0
        elif rc == 0:
            return 1
        else:
            return 0

    pitch = actions['camera'][:,0]
    yaw   = actions['camera'][:,1]

    rpitch = reconstructed_actions['camera'][:,0]
    ryaw   = reconstructed_actions['camera'][:,1]

    yaw_test = []
    pitch_test = []

    for y, ry, p, rp in zip (yaw,ryaw,pitch,rpitch):
        yaw_test.append(test_one_direction(y,ry))
        pitch_test.append(test_one_direction(y, ry))


    print("-------------------{}--------------".format('pitch'))
    print("{}% correctly restored.".format(np.sum(pitch_test)/len(pitch)))
    scores.append(np.sum(pitch_test)/len(pitch))

    print("-------------------{}--------------".format('yaw'))
    print("{}% correctly restored.".format(np.sum(yaw_test)/len(yaw)))
    scores.append(np.sum(yaw_test)/len(yaw))

    print("Total mean recosntruction:{}",np.mean(scores))



print("||||||||||||||||||||||||||||||maptozero={}||||||||||||||||||||||||||||||",True)
test(True)
print("||||||||||||||||||||||||||||||maptozero={}||||||||||||||||||||||||||||||",False)
test(False)

