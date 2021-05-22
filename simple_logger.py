import csv

#creating this class will overwrite given filename!
class SimpleLogger():

    def __init__(self,filename,colnames):
        self.colnames = colnames
        self.no_cols = len(colnames)
        self.filename=filename
        self.csv_file= open(filename, 'w', newline='\n')
        self.writer=csv.DictWriter(self.csv_file, colnames, delimiter=';')
        self.writer.writeheader()


    def log(self,values):
        assert len(values) == self.no_cols
        tuples = zip(self.colnames,values)
        self.writer.writerow(dict(tuples))

    def log_multiple(self,values_list):
        for values in values_list:
            self.log(values)

    def stop(self):
        self.csv_file.close()



