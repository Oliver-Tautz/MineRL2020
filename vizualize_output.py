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
print(acs[9])

# get active action names
aclist = []
for i, ac in enumerate(acs):
    ac_name = []
    for k,v in ac.items():
        if k == 'camera':
            if i == 12:
                print(v)
            if v[0][0] < 0:
                ac_name.append('pitch_negative')
            if v[0][0] >0:
                ac_name.append('pitch_positive')
            if v[0][1] < 0:
                    ac_name.append('yaw_negative')
            if v[0][1] > 0:
                    ac_name.append('yaw_positive')
        elif v[0] == 1:
            ac_name.append(k)
    aclist.append(ac_name)


print(list(enumerate(aclist)))
# concat names to use as labels for plot
aclist = [functools.reduce(lambda x, y: x +'_' +y,ac,'')[1:] for ac in aclist]


def visualize(pov,pred,label,prev_label,next_label):
    y = Softmax(0)(pred.squeeze().detach())
    width = 0.8



    x = np.arange(len(aclist))
    # y = np.random.sample(30)

    ## plotting
    plt.rcdefaults()
    # similar to tight_layout, but works better!
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)



    rects1 = ax2.barh(x, y,height=width/4,label='logits')

    y_ = np.zeros(30)
    y_[label[0]] = 1
    rects0 = ax2.barh(x-width/4, y_,label='label',height=width/4)


    if prev_label != None:
        y_ = np.zeros(30)
        y_[prev_label[0]] = 1
        ax2.barh(x+width / 4, y_,label='prev_label',height=width/4)


    if next_label != None:
        y_ = np.zeros(30)
        y_[next_label[0]] = 1
        ax2.barh(x-width/2, y_,label='next_label',height=width/4)



    plt.xlim(0, 1)

    ax1.imshow(pov.squeeze() / 255)
    ax1.axis('off')

    ax2.set_yticks(x)
    ax2.set_yticklabels(aclist, fontsize=3)
    ax2.tick_params(axis='x',labelsize=3,tickdir='in')
    ax2.set_xlabel('softmaxed logits',fontsize = 3)
    # This does not work well
    # plt.tight_layout()

   # plt.show()


model = Model(deviceStr='cpu', verbose=False, no_classes=30, with_masks=False)

model.load_state_dict(torch.load(
    "train/batch2_with-masks=False_map-to-zero=True_no-classes=30_seq-len=100_epoch=99_time=2021-06-09 18:14:29.514103.tm",
    map_location='cpu'))

train_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=1, map_to_zero=True,
                        with_masks=False, no_classes=30, no_replays=1)

model.eval()

states = model.get_zero_state(1)


# consider how many actions +-?
m = 2



for i in trange(1,len(train_set)-1):


    _, prev = train_set[i-1]
    pov,curr = train_set[i]
    _,next = train_set[i+1]


    pov_plot = torch.unsqueeze(pov, 0)
    pov = pov_plot.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

    pred, states = model.forward(pov, torch.zeros(64), states)

    visualize(pov_plot, pred, curr,prev,next)
    plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left',prop={'size': 5})
    plt.savefig(f"prediction_visualization/train_episode0/{i}.png",dpi=200)
    #plt.show()
    plt.close('all')
    plt.clf()

    if i%99==0:
        states = model.get_zero_state(1)





#fig, ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False)









