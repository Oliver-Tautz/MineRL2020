# Pytorch
import torch
import numpy as np
# Computer vison models using pytorch
import torchvision

# Transforming images
import torchvision.transforms as T

# Segmanetion wrapper/views
from .segm import Segmentation

# Segmentation dataset
class Dataset(torch.utils.data.Dataset):
    # Constructor
    def __init__( self, root, cmap, labels='mask', transform=T.Compose([]) ):
        # Dataset root directory
        self.root = root
        
        # Labels subdirectory
        self.labels = labels
        
        # Data transformation
        self.transform = transform
        
        # Colormap for transforming color coded segmentation to labels
        self.cmap = cmap
        
        # Listing files
        from os import walk
        
        # Find all files in root directory
        self.files = next(walk(root))[2]
        
    # Loads image and label from filename
    def load( self, filename ):
        # Filenames ans path processing
        from os.path import join
        
        # Image loading
        from PIL import Image
        
        # Load input image
        image = Image.open(join(self.root, filename))
        
        # Load label mask
        label = Image.open(join(self.root, self.labels, filename))
                      
        # Return pair of image+label
        return (self.transform(image),
            Segmentation.from_rgb(label,self.cmap).labels)
        
    # Get length of the dataset
    def __len__( self ):
        return len(self.files)
    
    # Get item from dataset at index
    def __getitem__( self, index ):
        # Load from filename at index
        return self.load(self.files[index])
        
    # Iterate all image+label pairs in dataset
    def __iter__( self ):
        # Iterate filenames
        for filename in self.files:
            # Load (image,label) pair
            yield self.load(filename)

# Segmentation dataset
class MineDataset(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, img_file, target_file, cmap, transform=T.Compose([])):
        # Dataset root directory
        self.img_file = img_file
        self.target_file = target_file
        self.cmap = cmap

        self.img_data = np.load(self.img_file)[:-4]
        self.target_data = np.asarray(np.load(self.target_file), dtype=np.uint8)[:-4]

        #self.load()
        # Data transformation
        self.transform = transform

    # Loads image and label from filename
    def load(self, index):
        # Filenames ans path processing
        # Return pair of image+label

        x = self.transform(self.img_data[index])
        y = Segmentation.from_rgb(self.target_data[index], self.cmap).labels
        return (x,y)

    # Get length of the dataset
    def __len__(self):
        return len(self.img_data)

    # Get item from dataset at index
    def __getitem__(self, index):
        # Load from filename at index
        return self.load(index)

    # Iterate all image+label pairs in dataset
    def __iter__(self):
        # Iterate filenames
        for index in range(len(self.img_data)):
            # Load (image,label) pair
            yield self.load(index)
