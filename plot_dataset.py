import os
import minerl
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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




no_streams = 30
actions = defaultdict(lambda : [])

# concats numpy arrays given in list.
# works

def one_hot(scalar_array,no_classes):
    return np.eye(no_classes)[scalar_array]

def in_for_np_array(array,list_of_arrays):
    in_list = False

    for arr in list_of_arrays:
        if np.array_equal(array,arr):
            in_list = True
    return in_list

def get_int_from_vector(int_to_array_dict,vector,zero_index):

    for i,vec in int_to_array_dict.items():
        if np.array_equal(vec,vector):
            return i

    print("Warning! Not yet seen action converted to zero_vector.")
    print(vector)
    return zero_index


# Takes dict of discrete actions from Treechop-v0 environment
# and returns one hot encoded actions with ndim = no_discrete_actions.
# If map_to_zero is True, infrequent actions will be mapped to the zero vector.

def transform_actions(actions, no_discrete_actions=30, map_to_zero=True, camera_noise_threshhold=0.5):
    pitch_positive, pitch_negative, yaw_positive, yaw_negative = discreticize_camera_action(actions['camera'],camera_noise_threshhold)

    actions['pitch_positive'] = pitch_positive
    actions['pitch_negative'] = pitch_negative
    actions['yaw_positive'] = yaw_positive
    actions['yaw_negative'] = yaw_negative
    
    no_sampled_actions = len(actions['pitch_positive'])
    actions.pop('camera')


    index_lookup = {key : ix for (ix,key) in enumerate(actions.keys())}
    key_lookup = dict(enumerate(actions.keys()))

    #print(index_lookup)


    # number of 0/1 values
    array_dimensionality = np.max(list(index_lookup.values()))+1
    #print('array_dimensionality: ', array_dimensionality)
    one_hot_encoding_size = 2**array_dimensionality
    #print('one_hot_dimensionality: ', one_hot_encoding_size)



    vectors = []
    for i in range(no_sampled_actions):
        action_vector = np.zeros(array_dimensionality)

        for key in actions.keys():
            action_vector[index_lookup[key]] = actions[key][i]

        vectors.append(action_vector)

    X = np.array(vectors)
    #print('means: ',np.mean(X,axis=0))

    unique, unique_counts = np.unique(X,axis=0,return_counts=True)

   # print(unique.shape)
    #print(unique_counts.shape)






    frequent_uniques_and_counts = list(sorted(zip(unique,unique_counts),key=lambda x: -x[1]))[0:no_discrete_actions]






    frequent_uniques = (list(map(lambda x:x[0],frequent_uniques_and_counts)))

    zero_vector = np.zeros(X[0].shape)
    zero_index = 0
    # make sure zero vector is in actions ...
    if not in_for_np_array(zero_vector,frequent_uniques):
        frequent_uniques.insert(zero_index,zero_vector)

    # print frequent actions...
    for action, count in frequent_uniques_and_counts:
        ixs= list((np.where(action==1)))[0]
        for i in ixs:
            print(key_lookup[i],'; ',end='')
        print('\t',count)

    mapped_vectors = []
    no_mapped = 0
    #print(frequent_uniques,X[0])


    # This is pretty slow! make it work in np?
    for x in X:
        if in_for_np_array(x,frequent_uniques):
            mapped_vectors.append(x)
        else:
            if map_to_zero:
                mapped_vectors.append(np.zeros(x.shape))
                no_mapped+=1
            # TODO implement map to closest action
            else:
                pass


    mapped_vectors=np.array(mapped_vectors)

    print("{}% of actions mapped".format(no_mapped/no_sampled_actions))

    int_to_vector_dict = dict(enumerate(frequent_uniques))

    integer_values = []

    for vec in mapped_vectors:
        integer_values.append(get_int_from_vector(int_to_vector_dict,vec,zero_index))

    print(len(integer_values))










def discreticize_camera_action(camera_actions,noise_threshhold):

    pitch = camera_actions[:,0]
    yaw   = camera_actions[:,1]

    print(pitch.shape)
   
    pitch_positive = pitch.copy()
    pitch_negative = pitch.copy()
    
    pitch_positive[pitch > noise_threshhold ] = 1
    pitch_positive[pitch <= noise_threshhold] = 0

    pitch_negative[pitch < -noise_threshhold ] = 1
    pitch_negative[pitch >= -noise_threshhold ] = 0

    yaw_positive = yaw.copy()
    yaw_negative = yaw.copy()

    yaw_positive[yaw > noise_threshhold] = 1
    yaw_positive[yaw <= noise_threshhold] = 0

    yaw_negative[yaw < -noise_threshhold] = 1
    yaw_negative[yaw >= -noise_threshhold] = 0

    print(np.sum(pitch_positive),np.sum(pitch_negative), np.sum(yaw_positive), np.sum(yaw_negative))

    return pitch_positive,pitch_negative, yaw_positive, yaw_negative


for f in (os.listdir('./data/MineRLTreechop-v0')):
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

transform_actions(actions)

