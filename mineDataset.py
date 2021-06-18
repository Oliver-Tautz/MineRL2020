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


    # there are 448480 unique steps in the dataset.


    def __init__(self,root_dir, sequence_length=100,with_masks=False,map_to_zero = True,cpus=3, no_replays = 300,no_classes=30,random_sequences=1000):



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


        self.mine_loader = minerl.data.make('MineRLTreechop-v0',data_dir=self.root_dir,num_workers=cpus)



        # full replay. Probably not needed
        # replays = dict()


        # length of replays in steps
        self.replays_length_raw=dict()

        # number of sequences in replay
        self.replays_length = dict()

        # pov of replay. Not batches in sequences
        self.replays_pov = dict()
        # act of replay as [int]. Not batches in sequences
        self.replays_act = dict()

        self.original_act = dict()
        for i, replay in tqdm(enumerate(self.replay_queue),total=len(self.replay_queue),desc=f'{self.message_str}loading replays'):
            d = self.mine_loader._load_data_pyfunc(os.path.join(self.root_dir,replay), -1, None)
            obs, act, reward, nextobs, done = d


            # -1 because we start at 0
            self.replays_length[i] = (len(obs['pov'])//sequence_length)-1
            self.replays_length_raw[i]  = len(obs['pov'])
            self.replays_pov[i] = torch.tensor(obs['pov'],dtype=torch.float32)
            self.original_act[i] = act
            self.replays_act[i] = torch.tensor(transform_actions(act,map_to_zero=map_to_zero,get_ints=True,no_classes=no_classes),dtype=torch.long)

            #replays[i] = d



        if random_sequences :
            self.random_sequences = []
            self.__randomize_sequences()

        self.len = sum(self.replays_length.values())

        if random_sequences:
            self.len = random_sequences

    def __print(self,str):
        print(f'{self.message_str}{str}')

    # returns (a,b) with a being the replay, and b being the start index in that replay.
    # Note: seems to be working
    def __map_ix_to_tuple(self,ix):

        if ix > self.len:
            self.__print('WARNING: Index too high!')

        i = 0

        # subtract len of replay from ix until we reach found replay

        while ix > 0:

            # wrap around?!
            #if i > max(self.replays_length.keys()):
            #    i=0
            if ix > self.replays_length[i]:
                ix-=self.replays_length[i]
                i+=1

            else:
                break

        return (i,ix)



    def __len__(self):
        return self.len

    def __getitem__(self,idx):

        if not self.random_sequences:
            replay_ix, sequence_ix = self.__map_ix_to_tuple(idx)

            start_ix = sequence_ix*self.sequence_length

            end_ix = start_ix+self.sequence_length


            pov = self.replays_pov[replay_ix][start_ix:end_ix]
            act = self.replays_act[replay_ix][start_ix:end_ix]

        else:
            pov,act = self.random_sequences[idx]

        return pov, act


    def __randomize_sequences(self):

        # sample from replay. Maybe also randomize this? Take random shuffling?
        replay_index = 0

        # number of replays -1
        last_replay_index = len(self.replay_queue)-1

        # get number of sequences
        for r_i in trange(self.no_random_sequences,desc='randomizing'):


            # roll random start
            sequence_start_index = np.random.randint(0,self.replays_length[replay_index])

            # reroll if too high ...
            while sequence_start_index+self.sequence_length >  self.replays_length_raw[replay_index]:
                sequence_start_index = np.random.randint(0, self.replays_length[replay_index])

            # append random sequence

            self.random_sequences.append((self.replays_pov[replay_index][sequence_start_index:sequence_start_index+self.sequence_length],
                                         self.replays_act[replay_index][sequence_start_index:sequence_start_index+self.sequence_length]))
            replay_index= (replay_index+1) % len(self.replay_queue)

        # delete unused stuff.
        del(self.replays_pov)
        del(self.replays_act)




if __name__ == '__main__':
    # test dataset.

    ds = MineDataset('data/MineRLTreechop-v0/train', no_replays=300, random_sequences=None, sequence_length=1)


    ds = MineDataset('data/MineRLTreechop-v0/train',no_replays=10,random_sequences=100,sequence_length=100)
    dataloader = DataLoader(ds, batch_size=1,
                        shuffle=False, num_workers=0,drop_last=True)

    recorder = EpisodeRecorder()


    for i, (pov ,_)  in enumerate(dataloader):
        for frame in pov.squeeze():
            recorder.record_frame(frame.numpy().astype(np.uint8))

        recorder.save_vid(f'dataset_vids/{i}.avi')
        recorder.reset()


