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
import argparse
from test import get_model_info_from_name
import os
import logging
from multiprocessing import Process, Lock

for name in [
             "matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('modelpath',help='use saved model at path',type=str)
parser.add_argument('--test-epochs',help='test every x epochs',type=int,default=4)
parser.add_argument('--no-classes',help="how many actions are there?",type=int,default=30)
parser.add_argument('--no-cpu',help="make torch use this number of threads",type=int,default=4)
parser.add_argument('--verbose',help="print more stuff",action="store_true")
parser.add_argument('--with-masks',help="use extra mask channel",action="store_true")
parser.add_argument('--save-vids',help="save videos of eval",action="store_true")
parser.add_argument('--max-steps',help="max steps per episode",type=int,default=500)
parser.add_argument('--sequence-len',help="reset states after how many steps?!",type=int,default=1000)
parser.add_argument('--num-threads',help="how many eval threads?",type=int,default=2)

args = parser.parse_args()



modelpath = args.modelpath
model_name = modelpath.split('/')[-1].split('.')[0]
no_classes = args.no_classes
no_cpu = args.no_cpu
verbose = args.verbose
with_masks=args.with_masks
save_vids = args.save_vids
max_steps=args.max_steps
test_epochs = args.test_epochs
num_threads = args.num_threads




torch.set_num_threads(10)
no_actions = 50

### setup stuff







def visualize(pov,pred,label,prev_label,next_label,no_classes):
    # get action representations

    acs = [transform_int_to_actions([x], camera_noise_threshhold=0.1, no_actions=no_actions) for x in range(no_classes)]

    # get active action names
    aclist = []
    for i, ac in enumerate(acs):
        ac_name = []
        for k, v in ac.items():
            if k == 'camera':
                if i == 12:
                    print(v)
                if v[0][0] < 0:
                    ac_name.append('pitch_negative')
                if v[0][0] > 0:
                    ac_name.append('pitch_positive')
                if v[0][1] < 0:
                    ac_name.append('yaw_negative')
                if v[0][1] > 0:
                    ac_name.append('yaw_positive')
            elif v[0] == 1:
                ac_name.append(k)
        aclist.append(ac_name)
    # concat names to use as labels for plot
    aclist = [functools.reduce(lambda x, y: x + '_' + y, ac, '')[1:] for ac in aclist]


    y = Softmax(0)(pred.squeeze().detach())
    #y = pred.squeeze().detach()
    #y[torch.argmax(y)] = 1
    #y[y<1] = 0
    print(y)
    width = 0.8



    x = np.arange(len(aclist))
    # y = np.random.sample(30)

    ## plotting
    plt.rcdefaults()
    # similar to tight_layout, but works better!
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    rects1 = ax2.barh(x, y,height=width/4,label='logits')

    y_ = np.zeros(no_classes)
    y_[label[0]] = 1
    rects0 = ax2.barh(x-width/4, y_,label='label',height=width/4)


    if prev_label != None:
        y_ = np.zeros(no_classes)
        y_[prev_label[0]] = 1
        ax2.barh(x+width / 4, y_,label='prev_label',height=width/4)


    if next_label != None:
        y_ = np.zeros(no_classes)
        y_[next_label[0]] = 1
        ax2.barh(x-width/2, y_,label='next_label',height=width/4)



    plt.xlim(0, 1)

    ax1.imshow(pov.squeeze())
    ax1.axis('off')

    ax2.set_yticks(x)
    ax2.set_yticklabels(aclist, fontsize=3)
    ax2.tick_params(axis='x',labelsize=3,tickdir='in')
    ax2.set_xlabel('softmaxed logits',fontsize = 3)
    # This does not work well
    # plt.tight_layout()

   # plt.show()








def visualize_output(modeldict):

    global no_classes
    no_classes = modeldict['no_classes']

    modelname = modeldict['name']

    model = Model(deviceStr='cpu', verbose=False, no_classes=modeldict['no_classes'], with_masks=modeldict['with_masks'],with_lstm=False)
    print(f'loading model {modelname}')
    model.load_state_dict(torch.load(os.path.join(modelpath,modeldict['name']),
        map_location='cpu'))
    print(f'loaded model {modelname}')
    train_set = MineDataset('data/MineRLTreechop-v0/train', sequence_length=1, map_to_zero=True,
                            with_masks=False, no_classes=modeldict['no_classes'], no_replays=1,random_sequences=None)

    model.eval()

    states = model.get_zero_state(1)

    for i in trange(1,len(train_set)-1):


        _, prev = train_set[i-1]
        pov,curr = train_set[i]
        _,next = train_set[i+1]


        pov_plot = torch.unsqueeze(pov, 0)
        pov = pov_plot.transpose(0, 1).transpose(2, 4).transpose(3, 4).contiguous()

        pred, states = model.forward(pov, torch.zeros(64), states)


        visualize(pov_plot, pred, curr,prev,next,modeldict['no_classes'])
        plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left',prop={'size': 5})
        os.makedirs(f"prediction_visualization/{modelname}",exist_ok=True)
        _ = plt.savefig(f"prediction_visualization/{modelname}/{i}.png",dpi=200)
        #plt.show()
        _  = plt.close('all')
        plt.clf()

        if i > max_steps :
            break
       # if i%99==0:
       #     states = model.get_zero_state(1)

threads = []
maxthreads = 6

for modelname_epoch in os.listdir(modelpath):

    if modelname_epoch.split('/')[-1].split('.')[-1] !='tm':
        continue

    modeldict = get_model_info_from_name(modelname_epoch)



    if not modeldict['epoch'] in range(10,20):
        continue



    p = Process(target=visualize_output,args=[modeldict])
   # visualize_output(modeldict)
    threads.append(p)
    p.start()

    if len(threads) >= maxthreads:
        print(f"waiting for {len(threads)} threads")
        for thread in threads:
            thread.join()

        threads = []

for thread in threads:
    thread.join()
    pass








