import matplotlib.pyplot as plt
import csv
import pandas as pd
from collections import defaultdict

class SimplePlotter():
    def __init__(self,csv_filename):
        self.csv_filename = csv_filename
        self.csv_file = open(csv_filename)
        self.csv_reader = csv.DictReader(self.csv_file,delimiter=';')
        #self.fig = plt.figure()

        self.data_dict = defaultdict(lambda: [])

        for row in self.csv_reader:
            for key in row.keys():
                self.data_dict[key].append(float(row[key]))

        self.df = pd.DataFrame(self.data_dict)



    def plot_line(self,colX,colY,startx=0):
        df = self.df.drop(range(0,startx))
        fig = df.plot(x=colX,y=colY,xlabel=colX,ylabel=colY)
        plt.show()
        plt.clf()





sp = SimplePlotter('loss_csv/working_closest.csv')
df = sp.df
