from torch.utils.data import Dataset, DataLoader
import torch
import minerl
import os
from tqdm import tqdm, trange
from descrete_actions_transform import transform_actions
from record_episode import EpisodeRecorder
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time
import sys
from functools import reduce


class MineDataset(Dataset):

    # root_dir          = path to minecraft replays
    # sequence_length   = length of sequences to be generated. Set < 1 to just get all (pov,act) pairs in the dataset.
    # with_masks        = load masks. need to be present!
    # map_to_zero       = true: map vectors to zero false: map vectors to closest ones
    # cpus              = mnake dataloader use multiple cores. Can not utilize more than 3 really
    # no_replays        = choose lower number of replays if ram is an issue.
    # random_sequences  = want randomized data? how many? set ==0 or None for sequential load.
    # return_float      = return RGB data normalized to 0-1
    # min_reward        = only use randoms sequences with at least this many rewards. Dataset has about ~2.8 reward mean in 100 steps, with 1.7 stddev
    # min_variance      = only use randoms sequences with at least this high action variance. Dataset has about ~41 reward mean in 100 steps, with ~20 stddev
    # max_verlap        = define max overlap for sequences. If defined, only so many random sequences are returned to not make them overlap more.

    # there are 448480 unique steps in the dataset.

    def __init__(self, root_dir, sequence_length=100, with_masks=False, map_to_zero=True, cpus=3, no_replays=300,
                 no_classes=30, random_sequences=50000, device='cuda', return_float=True, min_reward=0, min_variance=0,
                 max_overlap=10,ros=False,multilabel_actions=False,clean_samples=False):

        self.clean_samples=clean_samples

        self.max_overlap = max_overlap
        self.ros = ros
        self.multilabel_action = multilabel_actions

        # start with no overlap
        self.overlap_threshhold = sequence_length
        self.min_reward = min_reward
        self.min_variance = min_variance
        self.no_random_sequences = random_sequences

        self.map_to_zero = map_to_zero
        self.with_masks = with_masks

        # directory with files
        self.root_dir = root_dir

        # str to print before messages
        self.message_str = f'MineDataset[{self.root_dir}] : '

        # sequence length per sample
        self.sequence_length = sequence_length

        self.replay_queue = os.listdir(root_dir)[0:no_replays]

        self.mine_loader = minerl.data.make('MineRLTreechop-v0', data_dir=self.root_dir, num_workers=cpus)

        # full replay. Probably not needed
        # replays = dict()

        # length of replays in steps
        self.replays_length_raw = dict()

        # number of sequences in replay
        self.replays_length = dict()

        # pov of replay. Not batches in sequences
        self.replays_pov = dict()
        # act of replay as [int]. Not batches in sequences
        self.replays_act = dict()

        self.original_act = dict()
        self.replays_reward = dict()

        if with_masks:
            self.replays_masks = dict()

        for i, replay in tqdm(enumerate(self.replay_queue), total=len(self.replay_queue),
                              desc=f'{self.message_str}loading replays'):
            d = self.mine_loader._load_data_pyfunc(os.path.join(self.root_dir, replay), -1, None)
            obs, act, reward, nextobs, done = d

            # -1 because we start at 0
            self.replays_length[i] = (len(obs['pov']) // sequence_length) - 1
            self.replays_length_raw[i] = len(obs['pov'])

            if return_float:
                self.replays_pov[i] = torch.tensor(obs['pov'], dtype=torch.float32) / 255
            else:
                self.replays_pov[i] = torch.tensor(obs['pov'], dtype=torch.float32)

            self.original_act[i] = act
            if not multilabel_actions:
                self.replays_act[i] = torch.tensor(
                    transform_actions(act, map_to_zero=map_to_zero, get_ints=True, no_classes=no_classes,multilabel_actions=multilabel_actions), dtype=torch.long)
            else:
                self.replays_act[i] = torch.tensor(
                    transform_actions(act, map_to_zero=map_to_zero, get_ints=True, no_classes=no_classes,
                                      multilabel_actions=multilabel_actions), dtype=torch.float32)
            self.replays_reward[i] = reward

            if with_masks:
                self.replays_masks[i] = torch.load(os.path.join(self.root_dir, replay, 'masks.pt'), map_location='cpu')

            # replays[i] = d

        # compute reward mean and std.
        sums = []
        for reward in self.replays_reward.values():
            start = 0
            while start + 99 < len(reward):
                sums.append(sum(reward[start:start + 100]))
                start = start + 100

        print('reward mean = ', np.mean(sums))
        print('reward std  = ', np.std(sums))

        # compute action variance
        vars = []
        for acs in self.replays_act.values():
            start = 0
            while start + 99 < len(acs):
                vars.append(np.var(acs.numpy()[start:start + 100]))
                start = start + 100

        print('action_var mean = ', np.mean(vars))
        print('action_var std = ', np.std(vars))

        if sequence_length < 1:
            self.samples = []
            self.__init_samples()
            self.len = len(self.samples)
            return

        if random_sequences:
            self.random_sequences = []
            self.__randomize_sequences()
        else:
            self.random_sequences = None

        self.len = sum(self.replays_length.values())

        if random_sequences:
            self.len = len(self.random_sequences)


        if self.ros:
            self.len = len(self.ros_indexes)

    def __init_samples(self):
        for replay in range(len(self.replay_queue)):
            for i in range(self.replays_length_raw[replay]):
                if self.with_masks:
                    pov = torch.cat((self.replays_pov[replay][i], self.replays_masks[replay][i]), dim=-1)
                else:
                    pov = self.replays_pov[replay][i]
                self.samples.append((pov,self.replays_act[replay][i:i+1],self.replays_reward[replay][i]))

        del (self.replays_pov)
        del (self.replays_act)
        if self.with_masks:
            del (self.replays_masks)

        no_samples_without_cleaning = len(self.samples)
        cleaned = 0
        if self.clean_samples:
            for i, (pov,act,reward) in (reversed(list(enumerate(self.samples)))):
                rewards = [s[2] for s in self.samples[i-25:i+100]]
                rewardsum = sum(rewards)
                if rewardsum < self.min_reward:
                    self.samples.pop(i)
                    cleaned+=1
        self.__print(f'cleaned {cleaned} samples from dataset with {no_samples_without_cleaning} because vicinity reward < {self.min_reward}')

        if self.ros:
            # split dataset by class
            # this seems to work

            if self.multilabel_action:

                split_by_class = defaultdict(lambda : [])
                for i ,(_,act,_) in enumerate(self.samples):
                    for c in range(len(act.squeeze())):
                        if act.squeeze()[c] == 1:
                            split_by_class[c].append(i)



                counts = {x: len(y) for (x,y) in split_by_class.items() }

                biggest_class ,biggest_count = max(counts.items(),key=lambda x : x[1])
                smallest_class , smallest_count = min(counts.items(),key=lambda x : x[1])

                # oversample everything to half of most numerous class ...
                desired_count = int((biggest_count -smallest_count )*2/3)



                # remove samples with complex compound actions

                biggest_compound_cation = 4




                # this is only computed for small classes
                # oversampling pool will hold samples to  use in oversampling
                oversampling_pool = split_by_class.copy()


                for c in oversampling_pool.keys():
                    number_of_new_samples = desired_count - counts[c]

                    if number_of_new_samples <= 0 :
                        continue

                    ixs = []
                    for i,idx in enumerate(oversampling_pool[c]):
                        if self.samples[idx][1].sum()> biggest_compound_cation:
                            ixs.append(i)

                    #print('removed',len(ixs)/len(split_by_class[c])*100,f'% from class {c} for oversampling pool') ~10-20% of actions are complicated
                    for i in reversed(ixs):
                        oversampling_pool[c].pop(i)



                    all_class_samples_idx = oversampling_pool[c]
                    oversampling_pool[c] = defaultdict(lambda : [])

                    # compound number is the number of actions in a given compound action
                    for idx in all_class_samples_idx:
                        compound_number = self.samples[idx][1].sum().item()
                        oversampling_pool[c][compound_number].append(idx)


                # construct oversampled dataset

                for c in split_by_class.keys():
                    if desired_count - counts[c] <= 0:
                         continue

                    for compound_number in range(1,biggest_compound_cation):
                            sample_pool_size = len(oversampling_pool[c][compound_number])

                            # get new samples from pool, bu at max 3 times its own size!
                            number_of_samples_for_compound_level = min(int(number_of_new_samples*1/(2**compound_number)),3*sample_pool_size)

                            self.__print(f'chose {number_of_samples_for_compound_level} from {sample_pool_size} samples for class {c}')
                            split_by_class[c].extend(np.random.choice(oversampling_pool[c][compound_number], number_of_samples_for_compound_level))

                self.ros_indexes = np.random.permutation(reduce(lambda x, y: x + y, split_by_class.values(), []))
            else:

                class_indices = defaultdict(lambda :[])
                for i, ( _, act, _) in enumerate(self.samples):
                    class_indices[act[0].item()].append(i)

                class_counts =  dict()
                for key in class_indices.keys():
                    class_counts[key]=(len(class_indices[key]))



                # take middle off class counts as target sample size
                target_sample_size =    (max(class_counts.values())-min(class_counts.values()))//2

                for _class in class_indices.keys():

                    # oversample small classes
                    if class_counts[_class]<target_sample_size:
                        new_samples = np.random.choice(class_indices[_class],target_sample_size-class_counts[_class])
                        class_indices[_class].extend(new_samples)
                    else:
                        class_indices[_class]=class_indices[_class][0:target_sample_size]

                self.ros_indexes = np.random.permutation(reduce(lambda x,y : x+y,class_indices.values(),[]))



    def __print(self, str):
        print(f'{self.message_str}{str}')

    # returns (a,b) with a being the replay, and b being the start index in that replay.
    # Note: seems to be working
    def __map_ix_to_tuple(self, ix):

        if ix > self.len:
            self.__print('WARNING: Index too high!')

        i = 0

        # subtract len of replay from ix until we reach found replay

        while ix > 0:

            # wrap around?!
            # if i > max(self.replays_length.keys()):
            #    i=0
            if ix > self.replays_length[i]:
                ix -= self.replays_length[i]
                i += 1

            else:
                break

        return (i, ix)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.sequence_length < 0:

            if self.ros:
                idx = self.ros_indexes[idx]
                return self.samples[idx][0:2]
            else:
                # dont return reward ...
                return self.samples[idx][0:2]

        if not self.random_sequences:
            replay_ix, sequence_ix = self.__map_ix_to_tuple(idx)

            start_ix = sequence_ix * self.sequence_length

            end_ix = start_ix + self.sequence_length

            pov = self.replays_pov[replay_ix][start_ix:end_ix]
            act = self.replays_act[replay_ix][start_ix:end_ix]

            if self.with_masks:
                mask = self.replays_masks[replay_ix][start_ix:end_ix]
                pov = torch.cat((pov, mask), dim=-1)

        else:

            if self.with_masks:
                pov, act, mask = self.random_sequences[idx]
                pov = torch.cat((pov, mask), dim=-1)

            else:
                pov, act = self.random_sequences[idx]

        return pov, act

    def __randomize_sequences(self):

        # number of tries per replay for one sequence
        # 100 seems to work. take 300 to be sure.
        max_rejects_per_replay = 300

        # try this many times to get another sequence
        # try to get a sequence of each episode 3 more times
        max_rejects_on_max_threshold = len(self.replay_queue)*3

        used_indices = []

        # sample from replay. Maybe also randomize this? Take random shuffling?
        replay_index = 0

        # number of replays -1
        last_replay_index = len(self.replay_queue) - 1

        # get number of sequences
        rej = []
        r_i = 0
        max_overlap_rejection = 0

        self.__print('Generating ramdon sequences, this could take a while!')
        while r_i < self.no_random_sequences:

            if r_i %10 ==0:
                self.__print(f"produced {r_i} random sequences!{max_overlap_rejection}")

            too_high = True
            not_enough_reward = True
            not_enough_variance = True
            already_used = True

            rejected = -1

            while too_high or not_enough_reward or not_enough_variance or already_used:
                # print('rejected')
                rejected += 1
                # print('not enoug reward=', not_enough_reward)
                # print('not_enough_variance', not_enough_variance)

                sequence_start_index = np.random.randint(0, self.replays_length_raw[replay_index])

                # reroll if too high ...

                too_high = sequence_start_index + self.sequence_length > self.replays_length_raw[replay_index]

                # reroll if not enough reward

                not_enough_reward = sum(self.replays_reward[replay_index][
                                        sequence_start_index - 150:sequence_start_index + self.sequence_length + 150]) < self.min_reward

                # reroll if actions not diverse enough
                not_enough_variance = np.var(self.replays_act[replay_index][
                                             sequence_start_index:sequence_start_index + self.sequence_length].numpy()) < self.min_variance

                # print(too_high,'\t',sequence_start_index + self.sequence_length,'\n'
                #      ,not_enough_reward,'\t',sum(self.replays_reward[replay_index][sequence_start_index:sequence_start_index+self.sequence_length]),'\n'
                #      ,not_enough_variance,'\t',np.var(self.replays_act[replay_index][sequence_start_index:sequence_start_index+self.sequence_length].numpy()))

                def diff_in(tuple, list, threshhold):
                    ri, si = tuple

                    for ril, sil in list:
                        if ril == ri and abs(si - sil) <= threshhold:
                            return True

                    return False

                # reroll if sequence already used!
                already_used = diff_in((replay_index, sequence_start_index), used_indices, self.overlap_threshhold)

                # make sure it always works! lower overlap threshhold if too many sequences get rejected

                # print(rejected)

                if rejected >=max_rejects_per_replay:

                    current_overlap = self.sequence_length - self.overlap_threshhold
                    #print(current_overlap, self.max_overlap)
                    if current_overlap < self.max_overlap:
                        self.overlap_threshhold -= 1
                        self.__print(f'warning! overlap_threshold lowered to {self.overlap_threshhold - 1}, current max overlap is {current_overlap+1}')
                    else:
                        max_overlap_rejection += 1

                    if max_overlap_rejection > max_rejects_on_max_threshold:
                        break

                    replay_index = (replay_index + 1) % len(self.replay_queue)

                    too_high = True
                    not_enough_reward = True
                    not_enough_variance = True
                    already_used = True
                    rejected=-1

                    continue

            if max_overlap_rejection > max_rejects_on_max_threshold:
                break

            # print('accepted')
            rej.append(rejected)
            used_indices.append((replay_index, sequence_start_index))

            # append random sequence

            if self.with_masks:
                self.random_sequences.append(
                    (self.replays_pov[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_act[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_masks[replay_index][
                     sequence_start_index:sequence_start_index + self.sequence_length]))
                r_i += 1
            else:

                self.random_sequences.append(
                    (self.replays_pov[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_act[replay_index][sequence_start_index:sequence_start_index + self.sequence_length]))
                r_i += 1
                if len(self.random_sequences[-1][0]) < self.sequence_length:
                    self.__print(f"warning! sequence of length {len(self.random_sequences[-1][0])} produced")

            replay_index = (replay_index + 1) % len(self.replay_queue)

        # delete unused stuff.

        # print(used_indices)
        # print(sorted(used_indices,key=lambda x:x[0]))
        del (self.replays_pov)
        del (self.replays_act)
        if self.with_masks:
            del (self.replays_masks)





# [[0.43203875 0.01666584 0.4586578  0.0600334  0.06072781 0.05551973
#   0.01125936 0.05024552 0.19117768 0.25987467 0.19243424 0.21895409]]


if __name__ == '__main__':
    # test dataset.
    torch.set_printoptions(threshold=10_000)

    full_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=-1, map_to_zero=False,
                           with_masks=False, no_classes=12, no_replays=30,
                            device='cpu', ros=True,multilabel_actions=True,clean_samples=True,min_reward=4)

    acts = []
    for pov,act in full_set:
        acts.append(act)

    print(f"number of samples = {len(acts)}")
    acts= np.stack(acts)
    acts = np.mean(acts,axis =0)
    print(acts)
        #plt.show()
      #  plt.clf()
 #   for i, (pov, _, _) in enumerate(ds):
 #       # print(i)
 #       if len(pov) < 100:
 #           pass
 #           #print(len(pov))
 #
 #       # print(pov.shape)
 #       for frame in pov.squeeze():
 #           # print(frame.shape)
 #
 #           recorder.record_frame((frame * 255).numpy().astype(np.uint8))
 #           # masks_recorder.record_frame(mask.numpy().astype(np.uint8))
 #
 #       recorder.save_vid(f'dataset_vids/{i}.mp4')
 #
 #       recorder.reset()
 #       masks_recorder.reset()
 #       if i > 1000:
 #           break
 #