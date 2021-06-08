import torch
#import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import minerl


# Segmentation model
from pretrainedResnetMasks.segm.model import load_fcn_resnet101
from pretrainedResnetMasks.segm.data import MineDataset



class MaskGeneratorResnet():
    def __init__(self,device,model_filename='pretrainedResnetMasks/saved_res_lr_0.001/model.pt'):
        self.device =device
        self.model = load_fcn_resnet101(n_classes=8)
        self.model.load_state_dict(torch.load(model_filename, map_location=device))
        self.model = self.model.to(device)
        self.model = self.model.eval()

    def __prepare_img (self,img):
        IMG_SCALE = 1. / 255
        IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        return torch.tensor(((img * IMG_SCALE - IMG_MEAN) / IMG_STD).transpose(2, 0, 1), dtype=torch.float)

    def transform_image (self,img):
        prepped = self.__prepare_img(img)
        prepped= prepped.to(self.device).unsqueeze(0)
        mask = self.model(prepped)['out']
        mask = torch.argmax(mask.squeeze(), dim=0).detach().to(self.device).numpy()
        return mask

    def append_channel(self,img):
        mask = self.transform_image(img)
        mask = np.expand_dims(mask,axis=-1)

        combined = np.concatenate((img,mask),axis=2)
        return combined

    # input and output = torch.tensor :)
    # input = (batch,sequence,...)
    def append_channel_sequence_batch(self, batch):
        batch_shape = batch.shape

        # reshape to (batch,pic)
        reshaped = torch.reshape(batch,(batch_shape[0] * batch_shape[1], *batch_shape[2:]))

        # get masks from model
        masks = self.model(reshaped)['out']

        # postprocess to classes
        masks = torch.argmax(masks,dim=1)

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
        reshaped = torch.transpose(batch,1,3)


        # get masks from model
        masks = self.model(reshaped)['out']

        print(masks.shape)
        # postprocess to classes
        masks = torch.argmax(masks,dim=1)
        print(masks.shape)
        # add channel dimension
        masks = torch.unsqueeze(masks, dim=3)
        print(masks.shape)
        # concat channels
        masked = torch.cat((batch, masks), dim=3)


        return masked

#cmap = np.asarray([[0, 0, 0],
#                   [0, 0, 1],
#                   [0, 1, 0],
#                   [0, 1, 1],
#                   [1, 0, 0],
#                   [1, 0, 1],
#                   [1, 1, 0],
#                   [1, 1, 1]])
#
#
#
#img_file = 'pretrainedResnetMasks/data/X.npy'
#msk_file = 'pretrainedResnetMasks/data/Y.npy'
#dataset = MineDataset(img_file, msk_file, cmap)
#testfile = dataset[137][0]
#resnet = MaskGeneratorResnet('cpu')
#pred = resnet.transform_image(testfile)
#padded = resnet.append_channel(testfile)

# precompute masks for dataset.

def precompute_dir(filepath,device):
    loader = minerl.data.make('MineRLTreechop-v0',data_dir='./data',num_workers=1)
    resnet = MaskGeneratorResnet(device=device)

if __name__ == "__main__":
    loader = minerl.data.make('MineRLTreechop-v0',data_dir='./data',num_workers=1)
    resnet = MaskGeneratorResnet(device='cpu')
    print(os.listdir('./data/MineRLTreechop-v0/train'))
    f='v3_absolute_grape_changeling-15_10696-12887'

    d = loader._load_data_pyfunc('./data/MineRLTreechop-v0/train/{}'.format(f), -1, None)
    obs, act, reward, nextobs, done = d
    print(obs['pov'][0:100].shape)
    print(resnet.append_channel_batch(torch.tensor(obs['pov'][0:100], dtype=torch.float32)).shape)

    #for f in tqdm(os.listdir('./data/MineRLTreechop-v0/train'),desc='loading'):

