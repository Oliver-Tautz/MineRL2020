from torch.utils.data import Dataset, DataLoader
import torch
import minerl
import os
from tqdm import tqdm, trange
from descrete_actions_transform import transform_actions
from record_episode import EpisodeRecorder

import numpy as np
import time
import sys


class MineDataset(Dataset):

    # root_dir          = path to minecraft replays
    # sequence_length   = length of sequences to be generated
    # with_masks        = load masks. need to be present!
    # map_to_zero       = true: map vectors to zero false: map vectors to closest ones
    # cpus              = mnake dataloader use multiple cores. Can not utilize more than 3 really
    # no_replays        = choose lower number of replays if ram is an issue.
    # random_sequences  = want randomized data? how many? set ==0 or None for sequential load.
    # return_float      = return RGB data normalized to 0-1
    # min_reward        = only use randoms sequences with at least this many rewards. Dataset has about ~2.8 reward mean in 100 steps, with 1.7 stddev
    # min_variance      = only use randoms sequences with at least this high action variance. Dataset has about ~41 reward mean in 100 steps, with ~20 stddev

    # there are 448480 unique steps in the dataset.

    def __init__(self, root_dir, sequence_length=100, with_masks=False, map_to_zero=True, cpus=3, no_replays=300,
                 no_classes=30, random_sequences=1000,device='cuda',return_float = True,min_reward=0,min_variance=0,max_overlap=50):


        self.overlap_threshhold = sequence_length-max_overlap
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
                self.replays_pov[i] = torch.tensor(obs['pov'], dtype=torch.float32)/255
            else:
                self.replays_pov[i] = torch.tensor(obs['pov'], dtype=torch.float32)

            self.original_act[i] = act
            self.replays_act[i] = torch.tensor(

                transform_actions(act, map_to_zero=map_to_zero, get_ints=True, no_classes=no_classes), dtype=torch.long)
            self.replays_reward[i] = reward

            if with_masks:
                self.replays_masks[i] = torch.load(os.path.join(self.root_dir, replay, 'masks.pt'),map_location='cpu')

            # replays[i] = d

        # compute reward mean and std.
        sums = []
        for reward in self.replays_reward.values():
            start = 0
            while start+99<len(reward):
                sums.append(sum(reward[start:start+100]))
                start=start+100
        print('reward mean = ',np.mean(sums))
        print('reward std  = ',np.std(sums))

        # compute action variance
        vars = []
        for acs in self.replays_act.values():
            start = 0
            while start+99<len(acs):
                vars.append(np.var(acs.numpy()[start:start+100]))
                start=start+100

        print('action_var mean = ', np.mean(vars))
        print('action_var std = ',np.std(vars))




        if random_sequences:
            self.random_sequences = []
            self.__randomize_sequences()
        else:
            self.random_sequences = None

        self.len = sum(self.replays_length.values())

        if random_sequences:
            self.len = random_sequences

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

        if not self.random_sequences:
            replay_ix, sequence_ix = self.__map_ix_to_tuple(idx)

            start_ix = sequence_ix * self.sequence_length

            end_ix = start_ix + self.sequence_length

            pov = self.replays_pov[replay_ix][start_ix:end_ix]
            act = self.replays_act[replay_ix][start_ix:end_ix]

            if self.with_masks:

                mask = self.replays_masks[replay_ix][start_ix:end_ix]
                pov = torch.cat((pov,mask),dim=-1)

        else:

            if self.with_masks:
                pov, act, mask = self.random_sequences[idx]
                pov = torch.cat((pov,mask),dim=-1)

            else:
                pov, act = self.random_sequences[idx]



        return pov, act

    def __randomize_sequences(self):

        used_indices = []

        # sample from replay. Maybe also randomize this? Take random shuffling?
        replay_index = 0

        # number of replays -1
        last_replay_index = len(self.replay_queue) - 1

        # get number of sequences
        rej = []
        for r_i in trange(self.no_random_sequences, desc='randomizing'):

            too_high = True
            not_enough_reward = True
            not_enough_variance = True
            already_used = True


            rejected = -1
            while too_high or not_enough_reward or not_enough_variance or already_used:
                #print('rejected')
                rejected +=1
                #print('not enoug reward=', not_enough_reward)
                #print('not_enough_variance', not_enough_variance)


                sequence_start_index = np.random.randint(0, self.replays_length_raw[replay_index])

                # reroll if too high ...
                too_high = sequence_start_index + self.sequence_length > self.replays_length_raw[replay_index]


                # reroll if not enough reward
                not_enough_reward = sum(self.replays_reward[replay_index][sequence_start_index:sequence_start_index+self.sequence_length]) < self.min_reward

                # reroll if actions not diverse enough
                not_enough_variance = np.var(self.replays_act[replay_index][sequence_start_index:sequence_start_index+self.sequence_length].numpy()) < self.min_variance
                #print(too_high,'\t',sequence_start_index + self.sequence_length,'\n'
                #      ,not_enough_reward,'\t',sum(self.replays_reward[replay_index][sequence_start_index:sequence_start_index+self.sequence_length]),'\n'
                #      ,not_enough_variance,'\t',np.var(self.replays_act[replay_index][sequence_start_index:sequence_start_index+self.sequence_length].numpy()))

                def diff_in(tuple,list,threshhold):
                    ri, si = tuple

                    for ril,sil in list:
                        if ril == ri and abs(si - sil) <= threshhold:
                            return True

                    return False

                # reroll if sequence already used!
                already_used = diff_in((replay_index,sequence_start_index),used_indices,self.overlap_threshhold)

                # make sure it always works!

                if rejected >= 100:
                    print('warning! overlap_threwshold lowered')
                    self.overlap_threshhold-=1
                    break



            #print('accepted')
            rej.append(rejected)
            used_indices.append((replay_index,sequence_start_index))

            # append random sequence



            if self.with_masks:
                self.random_sequences.append(
                    (self.replays_pov[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_act[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_masks[replay_index][sequence_start_index:sequence_start_index + self.sequence_length]))
            else:

                self.random_sequences.append(
                    (self.replays_pov[replay_index][sequence_start_index:sequence_start_index + self.sequence_length],
                     self.replays_act[replay_index][sequence_start_index:sequence_start_index + self.sequence_length]))

            replay_index = (replay_index + 1) % len(self.replay_queue)

        # delete unused stuff.

        #print(sorted(used_indices,key=lambda x:x[0]))
        del (self.replays_pov)
        del (self.replays_act)
        if self.with_masks:
            del (self.replays_masks)


if __name__ == '__main__':
    # test dataset.
    torch.set_printoptions(threshold=10_000)

    ds = MineDataset('data/MineRLTreechop-v0/train', no_replays=10, random_sequences=100, sequence_length=100,device='cpu',with_masks=False,min_reward=2,min_variance=30)


    dataloader = DataLoader(ds, batch_size=1,
                            shuffle=True, num_workers=0, drop_last=True)

    recorder = EpisodeRecorder()
    masks_recorder = EpisodeRecorder()

    for i, (pov, _) in enumerate(dataloader):
        #print(pov.shape)
        for frame in  pov.squeeze():
            #print(frame.shape)

            recorder.record_frame((frame*255).numpy().astype(np.uint8))
            #masks_recorder.record_frame(mask.numpy().astype(np.uint8))

        recorder.save_vid(f'dataset_vids/{i}.mp4')

        recorder.reset()
        masks_recorder.reset()
