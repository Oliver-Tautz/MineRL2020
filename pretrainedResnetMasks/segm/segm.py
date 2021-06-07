# Math, Arrays, ...
import numpy as np

# Segmentation class, provides different views/interpretations of segmentation
class Segmentation(object):
    # Constructor
    def __init__( self, cmap, labels ):
        # Segmentation color coding
        self.cmap = cmap
        
        # Segmentation labels
        self.labels = labels
        
        # Number of segmanetaion classes
        self.n_classes = len(cmap)
        
    # Constructs segmentation from rgb color coding
    @staticmethod
    def from_rgb( rgb, cmap ):
        # Convert to numpy
        rgb = np.array(rgb)
        
        # Construct inverse of the colormap
        cmap_inv = {tuple(key): value for value, key in enumerate(cmap)}

        # Create zero initialized array of size of the input image
        mapped = np.zeros(shape=(rgb.shape[0],rgb.shape[1]))

        # Iterate over all pixels
        for x in range(rgb.shape[0]):
            for y in range(rgb.shape[1]):
                # Map pixel using colormap
                mapped[x,y] = cmap_inv[tuple(rgb[x,y,:])]

        # Wrap in Segmentation object
        return Segmentation(cmap=cmap, labels=mapped.astype(np.uint8))
    
    # Gets segmentation as color coded rgb
    @property
    def rgb( self ):
        return self.cmap[np.array(self.labels).astype(np.uint8)]
    
    # Gets stack of segmentation mask per class
    @property
    def stack( self ):
        # Create array of image size but a channel per class
        stack = []
        
        # For each class
        for c in range(self.n_classes):
            # Mask all pixels labeled as class
            stack.append((self.labels == c).astype(np.uint8))
            
        # Convert to numpy array
        return np.stack(stack)
    
    # Gets stack of segmentation mask per class arranges horizontally
    @property
    def hstack( self ):
        # Create array of image size but a channel per class
        stack = []
        
        # For each class
        for c in range(self.n_classes):
            # Mask all pixels labeled as class
            stack.append((self.labels == c).astype(np.uint8))
            
        # Convert to numpy array
        return np.hstack(stack) 
