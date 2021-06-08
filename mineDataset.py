from torch.utils.data import Dataset, DataLoader
import torch
import minerl
import os
from tqdm import tqdm
from descrete_actions_transform import transform_actions

import time
import sys



class MineDataset(Dataset):

    # root_dir          = path to minecraft replays
    # sequence_length   = length of sequences to be generated
    # with_masks        = load masks. need to be present!
    # map_to_zero       = true: map vectors to zero false: map vectors to closest ones
    # cpus              = mnake dataloader use multiple cores. Can not utilize more than 3 really
    # no_replays        = choose lower number of replays if ram is an issue.

    def __init__(self,root_dir, sequence_length=100,with_masks=False,map_to_zero = True,cpus=3, no_replays = 300,no_classes=30):




        self.map_to_zero = map_to_zero
        self.with_masks = with_masks

        # directory with files
        self.root_dir = root_dir

        # str to print before messages
        self.message_str = f'MineDataset[{self.root_dir}] : '

        # sequence length per sample
        self.sequence_length = sequence_length

        # replay queue:       data still to sample in epoch
        # already used queue: data already sampled in epoch
        # both just carry the string names of folders.

        self.replay_queue = os.listdir(root_dir)[0:no_replays]

        self.mine_loader = minerl.data.make('MineRLTreechop-v0',data_dir=self.root_dir,num_workers=cpus)



        # full replay. Probably not needed
        #replays = dict()

        # number of sequences in replay
        self.replays_length = dict()

        # pov of replay. Not batches in sequences
        self.replays_pov = dict()
        # act of replay as [int]. Not batches in sequences
        self.replays_act = dict()
        
        for i, replay in tqdm(enumerate(self.replay_queue),total=len(self.replay_queue),desc=f'{self.message_str}loading replays'):
            d = self.mine_loader._load_data_pyfunc(os.path.join(self.root_dir,replay), -1, None)
            obs, act, reward, nextobs, done = d


            # -1 because we start at 0
            self.replays_length[i] = (len(obs['pov'])//sequence_length)-1

            self.replays_pov[i] = torch.tensor(obs['pov'],dtype=torch.float32)
            self.replays_act[i] = torch.tensor(transform_actions(act,map_to_zero=map_to_zero,get_ints=True,no_classes=no_classes),dtype=torch.long)
            #replays[i] = d



        self.len = sum(self.replays_length.values())


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
            if ix > self.replays_length[i]:
                ix-=self.replays_length[i]
                i+=1

            else:
                break

        return (i,ix)



    def __len__(self):
        return self.len

    def __getitem__(self,idx):

        replay_ix, sequence_ix = self.__map_ix_to_tuple(idx)

        start_ix = sequence_ix*self.sequence_length

        end_ix = start_ix+self.sequence_length


        pov = self.replays_pov[replay_ix][start_ix:end_ix]
        act = self.replays_act[replay_ix][start_ix:end_ix]

        return pov, act







if __name__ == '__main__':
    ds = MineDataset('data/MineRLTreechop-v0/val')
    dataloader = DataLoader(ds, batch_size=4,
                        shuffle=True, num_workers=0)
