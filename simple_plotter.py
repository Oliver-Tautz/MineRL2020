import matplotlib.pyplot as plt
import csv
import pandas as pd
from collections import defaultdict
import os
from functools import reduce

class SimplePlotter():
    def __init__(self,csv_filename):
        self.csv_filename = csv_filename
        self.csv_file = open(csv_filename)
        self.csv_reader = csv.DictReader(self.csv_file,delimiter=';')
        #self.fig = plt.figure()

        self.data_dict = defaultdict(lambda: [])

        for row in self.csv_reader:
            for key in row.keys():
                try :
                    self.data_dict[key].append(float(row[key]))
                except:
                    self.data_dict[key].append(str(row[key]))

        self.df = pd.DataFrame(self.data_dict)
        self.checkpoint_x = x = [350000/20*i for i in range(1,20)]



    def plot_line(self,colX,colY,startx=0):
        #print(self.df)
        if len(self.df) == 0:
            return
        df = self.df.drop(range(0,startx))


        fig = df.plot(x=colX,y=colY,xlabel=colX,ylabel=colY)

    def show(self):
        plt.show()

    def clear(self):
        plt.clf()

    def plot_vlines(self,x):
        plt.vlines(x,linestyles = 'dotted',  color="green",ymin=0,ymax=3,label='checkpoints')

    def set_title(self,title):
        plt.title(title)

#$#sp = SimplePlotter('loss_csv/working_closest.csv')

def wrap_string(string,max_len):
    l = 0
    substrings = []
    while l < len(string):
       # print(l,len(string))
        substrings.append(string[l:l+max_len])
        l+=max_len

    #print(substrings)
    return reduce(lambda x,y: x+'\n'+y,substrings,'')

batchdir = '/home/olli/remote/techfak/compute/gits/MineRL2020/train'

for batch in os.listdir(batchdir):
    for file in os.listdir(f'{batchdir}/{batch}'):
        if '.csv' in file  and 'lock' not in file:
            print(f'plotting {file}')
            sp = SimplePlotter(f'{batchdir}/{batch}/{file}')
            sp.plot_line('epoch',['loss','val_loss'])
            filename = file.split('.csv')[0]
            sp.set_title(wrap_string(filename,50))
            plt.savefig(f'{batchdir}/{batch}/{filename}_loss.pdf')
            plt.clf()

            sp.plot_line('epoch',['acc','val_acc'])
            filename = file.split('.csv')[0]
            sp.set_title(wrap_string(filename,50))
            plt.savefig(f'{batchdir}/{batch}/{filename}_acc.pdf')
            plt.clf()

