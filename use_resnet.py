import torch
#import matplotlib.pyplot as plt
import numpy as np



# Segmentation model
from pretrainedResnetMasks.segm.model import load_fcn_resnet101
from pretrainedResnetMasks.segm.data import MineDataset



class MaskGeneratorResnet():
    def __init__(self,device):
        self.device =device
        self.model = load_fcn_resnet101(n_classes=8)
        self.model.load_state_dict(torch.load('/home/olli/gits/MichalOpMineRL2020/pretrainedResnetMasks/saved_res_lr_0.001/model.pt', map_location=device))
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

    def append_channel_batch(self,batch):
        pass


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



