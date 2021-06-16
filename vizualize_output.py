import minerl
from model import Model
from mineDataset import MineDataset
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt
from descrete_actions_transform import load_obj, transform_int_to_actions
import functools
import numpy as np
from torch.nn import Softmax





torch.set_num_threads(10)


### setup stuff

# get action representations
acs = [transform_int_to_actions([x],camera_noise_threshhold=0.1,) for x in range(30)]

# get active action names
aclist = []
for ac in acs:
    ac_name = []
    for k,v in ac.items():
        if k == 'camera':
            if v[0][0] < 0:
                ac_name.append('pitch_negative')
            elif v[0][0] >0:
                ac_name.append('pitch_positive')

                if v[0][1] < 0:
                    ac_name.append('yaw_negative')
                elif v[0][1] > 0:
                    ac_name.append('yaw_positive')
        elif v[0] == 1:
            ac_name.append(k)
    aclist.append(ac_name)


# concat names to use as labels for plot
aclist = [functools.reduce(lambda x, y: x +'_' +y,ac,'')[1:] for ac in aclist]


def visualize(pov,pred,label):
    y = Softmax(0)(pred.squeeze().detach())
    y_ = np.zeros(30)
    y_[label[0]] = 0.05

    x = aclist
    # y = np.random.sample(30)

    ## plotting
    plt.rcdefaults()
    # similar to tight_layout, but works better!
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    ax2.barh(x, y)
    ax2.barh(x, y_, left=y)

    plt.xlim(0, 1)
    ax1.imshow(pov.squeeze() / 255)
    ax1.axis('off')
    # This does not work well
    # plt.tight_layout()

   # plt.show()


model = Model(deviceStr='cpu', verbose=False, no_classes=30, with_masks=False)

model.load_state_dict(torch.load(
    "train/batch2_with-masks=False_map-to-zero=True_no-classes=30_seq-len=100_epoch=99_time=2021-06-09 18:14:29.514103.tm",
    map_location='cpu'))

train_set = MineDataset('data/MineRLTreechop-v0/val', sequence_length=1, map_to_zero=True,
                        with_masks=False, no_classes=30, no_replays=1)

model.eval()

states = model.get_zero_state(1)



for i in trange(len(train_set)):
    pov,action = train_set[i]

    pov_plot = torch.unsqueeze(pov, 0)
    pov = pov_plot.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

    hidden, pred, states = model.compute_front(pov, torch.zeros(64), states)

    visualize(pov_plot, pred, action)
    plt.savefig(f"prediction_visualization/eval_episode0/{i}.png",dpi=200)
    plt.close('all')
    plt.clf()

    if i%99==0:
        states = model.get_zero_state(1)





#fig, ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False)









