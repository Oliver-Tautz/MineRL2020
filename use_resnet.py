import torch
# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import minerl

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Segmentation model
from pretrainedResnetMasks.segm.model import load_fcn_resnet101
from pretrainedResnetMasks.segm.data import MineDataset
from pretrainedResnetMasks.segm.segm import Segmentation




class MaskGeneratorResnet():
    def __init__(self, device, model_filename='pretrainedResnetMasks/saved_res_lr_0.001/model.pt'):
        self.device = device
        self.model = load_fcn_resnet101(n_classes=8)
        self.model.load_state_dict(torch.load(model_filename, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.model.double()


    def __prepare_img(self, img):
        IMG_SCALE = 1. / 255
        IMG_MEAN = torch.tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)))
        IMG_STD = torch.tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
        return ((img * IMG_SCALE - IMG_MEAN) / IMG_STD).transpose(0,2).transpose(1,2)

    def __prepare_batch(self, batch):
        IMG_SCALE = torch.tensor(1. / 255,device=self.device)
        IMG_MEAN = torch.tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),device=self.device)
        IMG_STD = torch.tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),device=self.device)
        return ((batch * IMG_SCALE - IMG_MEAN) / IMG_STD).transpose(1,3).transpose(2,3)

    def transform_image(self, img):
        prepped = self.__prepare_img(img)
        prepped = prepped.to(self.device).unsqueeze(0)
        print(prepped)
        mask = self.model(prepped)['out']
        mask = torch.argmax(mask.squeeze(), dim=0).detach().to(self.device)
        return mask

    def append_channel(self, img):
        mask = self.transform_image(img)
        mask = np.expand_dims(mask, axis=-1)

        combined = np.concatenate((img, mask), axis=2)
        return combined

    # input and output = torch.tensor :)
    # input = (batch,sequence,...)
    def append_channel_sequence_batch(self, batch):
        batch_shape = batch.shape

        # reshape to (batch,pic)
        reshaped = torch.reshape(batch, (batch_shape[0] * batch_shape[1], *batch_shape[2:]))

        # get masks from model
        masks = self.model(reshaped)['out']

        # postprocess to classes
        masks = torch.argmax(masks, dim=1)

        # reshape back to original shape
        masks = torch.reshape(masks, (batch_shape[0], batch_shape[1], *masks.shape[1:]))

        # add channel dimension
        masks = torch.unsqueeze(masks, dim=2)

        # concat channels
        masked = torch.cat((batch, masks), dim=2)

        return masked

    # input and output = torch.tensor :)
    # input = (batch,...)

    def append_channel_batch(self, batch):
        batch_shape = batch.shape

        # reshape to (batch,pic)
        reshaped = torch.transpose(batch, 1, 3)

        # get masks from model
        masks = self.model(reshaped)['out']

        print(masks.shape)
        # postprocess to classes
        masks = torch.argmax(masks, dim=1)
        print(masks.shape)
        # add channel dimension
        masks = torch.unsqueeze(masks, dim=3)
        print(masks.shape)
        # concat channels
        masked = torch.cat((batch, masks), dim=3)

        return masked

    # batch = (batch,x,y,c)
    def return_masks(self, batch):
        batch_shape = batch.shape
        print(batch_shape)

        prepped = self.__prepare_batch(batch)
        #print(reshaped.shape)
        # get masks from model
        masks = self.model(prepped)['out']


        # postprocess to classes
        masks = torch.argmax(masks, dim=1)

        # add channel dimension
        masks = torch.unsqueeze(masks, dim=3)

        return masks





def precompute_dir(filepath, device,batchsize=10):
    resnet = MaskGeneratorResnet(device=device)
    loader = minerl.data.make('MineRLTreechop-v0', data_dir='./data', num_workers=4)

    for replay in tqdm(os.listdir(filepath)[0:1], desc='loading'):
        full_name = os.path.join(filepath, replay)

        d = loader._load_data_pyfunc(full_name, -1, None)
        obs, act, reward, nextobs, done = d


        dl = DataLoader(dataset= torch.tensor(obs['pov'],dtype=torch.float32,device=device),batch_size=batchsize,shuffle=False,num_workers=0,drop_last=False,pin_memory=False)


        masks = []
        for batch in tqdm(dl,desc=f'precomputing masks for dir {full_name}',position=0 , leave=True):

            masks.append(resnet.return_masks(batch))

        masks=torch.cat(masks,dim=0)

        torch.save(masks,os.path.join(full_name,'masks.pt'))







    #torch.save(masks, os.path.join(filepath, replay, 'mask.pt'))


if __name__ == '__main__':
    deviceStr = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('using device:', device)

    #torch.set_num_threads(10)

    precompute_dir('./data/MineRLTreechop-v0/train',device=deviceStr)




