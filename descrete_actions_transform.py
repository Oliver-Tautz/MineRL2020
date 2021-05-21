import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# This dict maps integers to actions in 0/1 encoding
int_to_vec_filename = 'int_to_vec_dict'

# This dict maps the actions keys to positions in the vectors in 0/1 encoding
key_to_index_filename = 'key_to_index_dict'

# Movement below this point will be regarded as no camera action.
camera_noise_threshhold = 0.5


# save to pkl
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# load from pkl
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Takes dict of discrete actions from Treechop-v0 environment
# and returns one hot encoded actions with ndim = no_discrete_actions.
# If map_to_zero is True, infrequent actions will be mapped to the zero vector.


def save_frequent_actions_and_mapping(actions, no_discrete_actions=30, camera_noise_threshhold=camera_noise_threshhold):
    actions = actions.copy()

    pitch_positive, pitch_negative, yaw_positive, yaw_negative = discreticize_camera_action(actions['camera'],
                                                                                            camera_noise_threshhold)

    actions['pitch_positive'] = pitch_positive
    actions['pitch_negative'] = pitch_negative
    actions['yaw_positive'] = yaw_positive
    actions['yaw_negative'] = yaw_negative

    no_sampled_actions = len(actions['pitch_positive'])
    actions.pop('camera')

    index_lookup = {key: ix for (ix, key) in enumerate(actions.keys())}
    key_lookup = dict(enumerate(actions.keys()))

    # number of 0/1 values
    array_dimensionality = np.max(list(index_lookup.values())) + 1
    print('array_dimensionality: ', array_dimensionality)
    one_hot_encoding_size = 2 ** array_dimensionality
    # print('one_hot_dimensionality: ', one_hot_encoding_size)

    vectors = []
    for i in tqdm(range(no_sampled_actions), desc='get_action_vectors'):
        action_vector = np.zeros(array_dimensionality)

        for key in actions.keys():
            action_vector[index_lookup[key]] = actions[key][i]

        vectors.append(action_vector)

    X = np.array(vectors)
    # print('means: ',np.mean(X,axis=0))

    unique, unique_counts = np.unique(X, axis=0, return_counts=True)

    # print(unique.shape)
    # print(unique_counts.shape)

    frequent_uniques_and_counts = list(sorted(zip(unique, unique_counts), key=lambda x: -x[1]))[0:no_discrete_actions]

    frequent_uniques = (list(map(lambda x: x[0], frequent_uniques_and_counts)))

    zero_vector = np.zeros(X[0].shape)
    zero_index = 0
    # make sure zero vector is in actions ...
    if not in_for_np_array(zero_vector, frequent_uniques):
        frequent_uniques.insert(zero_index, zero_vector)

    # print frequent actions...
    for action, count in frequent_uniques_and_counts:
        ixs = list((np.where(action == 1)))[0]
        action_string = ""
        for i in ixs:
            action_string += key_lookup[i] + ' & '
        print("{:70s}\t{:5d}".format(action_string, count))

    int_to_vector_dict = dict(enumerate(frequent_uniques))

    save_obj(int_to_vector_dict, int_to_vec_filename)
    save_obj(key_lookup, key_to_index_filename)

    return int_to_vector_dict, key_lookup


def transform_onehot_to_actions(X,camera_noise_threshhold=camera_noise_threshhold):

    key_lookup = load_obj(key_to_index_filename)
    int_to_vector_dict = load_obj(int_to_vec_filename)

    actions = defaultdict(lambda :[])
    ints = onehot_to_int(X)

    vecs = []
    for i in ints:
        vecs.append(int_to_vector_dict[i])


    for vec in vecs:
        for index in key_lookup.keys():
            if vec[index] == 1:
                actions[key_lookup[index]].append(1)
            elif vec[index] ==0:
                 actions[key_lookup[index]].append(0)
            else:
                actions[key_lookup[index]].append(0)
                print("Warning! Trying to convert non binary values to actions.")

    
    pitch_positive = actions.pop('pitch_positive')
    pitch_negative = actions.pop('pitch_negative')
    yaw_positive = actions.pop('yaw_positive')
    yaw_negative = actions.pop('yaw_negative')
    

    pitch_action = []

    
    for pos, neg in zip(pitch_positive,pitch_negative):
        if pos and neg:
            print("Warning, both camera directions at the same timestep!")
            pitch_action.append(2)
        elif pos:
            pitch_action.append(2)
        elif neg:
            pitch_action.append(-2)
        else:
            pitch_action.append(0)

    yaw_action = []

    for pos, neg in zip(yaw_positive, yaw_negative):
        if pos and neg:
            print("Warning, both camera directions at the same timestep!")
            yaw_action.append(2)
        elif pos:
            yaw_action.append(2)
        elif neg:
            yaw_action.append(-2)
        else:
            yaw_action.append(0)

    pitch_action = np.array(pitch_action)
    yaw_action = np.array(yaw_action)

    camera_action = np.column_stack((pitch_action,yaw_action))
    actions['camera']=camera_action

    for key in actions.keys():
        actions[key] = np.array(actions[key])


    return actions




def transform_actions_to_onehot(actions, map_to_zero=True, camera_noise_threshhold=camera_noise_threshhold):
    pitch_positive, pitch_negative, yaw_positive, yaw_negative = discreticize_camera_action(actions['camera'],
                                                                                            camera_noise_threshhold)


    actions = actions.copy()
    key_lookup = load_obj(key_to_index_filename)
    int_to_vector_dict = load_obj(int_to_vec_filename)

    actions['pitch_positive'] = pitch_positive
    actions['pitch_negative'] = pitch_negative
    actions['yaw_positive'] = yaw_positive
    actions['yaw_negative'] = yaw_negative

    no_sampled_actions = len(actions['pitch_positive'])
    actions.pop('camera')


    index_lookup = {key: ix for (ix, key) in key_lookup.items()}

    # print(index_lookup)

    # number of 0/1 values
    array_dimensionality = np.max(list(index_lookup.values())) + 1
    print('array_dimensionality: ', array_dimensionality)
    one_hot_encoding_size = 2 ** array_dimensionality
    # print('one_hot_dimensionality: ', one_hot_encoding_size)

    vectors = []
    for i in (range(no_sampled_actions)):
        action_vector = np.zeros(array_dimensionality)

        for key in actions.keys():
            action_vector[index_lookup[key]] = actions[key][i]

        vectors.append(action_vector)

    X = np.array(vectors)
    # print('means: ',np.mean(X,axis=0))


    no_discrete_actions = len(int_to_vector_dict.values())
    frequent_uniques = list(int_to_vector_dict.values())

    zero_vector = np.zeros(X[0].shape)
    zero_index = 0
    # make sure zero vector is in actions ...
    if not in_for_np_array(zero_vector, frequent_uniques):
        frequent_uniques.insert(zero_index, zero_vector)

    mapped_vectors = []
    no_mapped = 0
    # print(frequent_uniques,X[0])

    # This is pretty slow! make it work in np?
    for x in (X):
        if in_for_np_array(x, frequent_uniques):
            mapped_vectors.append(x)
        else:
            if map_to_zero:
                mapped_vectors.append(np.zeros(x.shape))
                no_mapped += 1
            # TODO implement map to closest action
            else:
                no_mapped += 1
                smallest_dist = 1000000
                for i,unique in enumerate(frequent_uniques):
                    dist = np.sum(np.logical_not(np.logical_xor(x,unique)))
                    if dist < smallest_dist:
                        smallest_dist = dist
                        smallest_ix = i
                mapped_vectors.append(frequent_uniques[smallest_ix])


    mapped_vectors = np.array(mapped_vectors)

    print("{}% of actions mapped".format(no_mapped / no_sampled_actions))

    int_to_vector_dict = dict(enumerate(frequent_uniques))

    integer_values = []

    for vec in (mapped_vectors):
        integer_values.append(get_int_from_vector(int_to_vector_dict, vec, zero_index))

    return int_to_one_hot(integer_values, no_discrete_actions)




def discreticize_camera_action(camera_actions, noise_threshhold):
    pitch = camera_actions[:, 0]
    yaw = camera_actions[:, 1]

    pitch_positive = pitch.copy()
    pitch_negative = pitch.copy()

    pitch_positive[pitch > noise_threshhold] = 1
    pitch_positive[pitch <= noise_threshhold] = 0

    pitch_negative[pitch < -noise_threshhold] = 1
    pitch_negative[pitch >= -noise_threshhold] = 0

    yaw_positive = yaw.copy()
    yaw_negative = yaw.copy()

    yaw_positive[yaw > noise_threshhold] = 1
    yaw_positive[yaw <= noise_threshhold] = 0

    yaw_negative[yaw < -noise_threshhold] = 1
    yaw_negative[yaw >= -noise_threshhold] = 0

    return pitch_positive, pitch_negative, yaw_positive, yaw_negative



def int_to_one_hot(scalar_array, no_classes):
    return np.eye(no_classes)[scalar_array]

def onehot_to_int(onehot_array):
    return np.argmax(onehot_array,axis=1)

def in_for_np_array(array, list_of_arrays):
    in_list = False

    for arr in list_of_arrays:
        if np.array_equal(array, arr):
            in_list = True
    return in_list


def get_int_from_vector(int_to_array_dict, vector, zero_index):
    for i, vec in int_to_array_dict.items():
        if np.array_equal(vec, vector):
            return i

    print("Warning! Not yet seen action converted to zero_vector.")
    print(vector)
    return zero_index
