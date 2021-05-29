# Pytorch
import torch

# Computer vison models using pytorch
import torchvision

# Reconfigures segmantation model (backbone) with diffrent number of classifier
# output channels
class ReconfigureClassifier(torchvision.models.segmentation.fcn.FCN):
    # Constructor reconfigures backbone
    def __init__( self, model, n_classes ):
        # Last layer (the classifier), fully convolutional
        from torchvision.models.segmentation.fcn import FCNHead
        
        # Modify classifier layer to predict requested number of classes
        classifier = FCNHead(
            model.classifier[0].in_channels, n_classes)
        
        # Modify aux classifier layer to predict requested number of classes
        aux_classifier = FCNHead(
            model.aux_classifier[0].in_channels, n_classes)
        
        # Call Module constructor to initialize network module
        super(torchvision.models.segmentation.fcn.FCN, self).__init__(
            model.backbone, classifier, aux_classifier)
        
# Loads trained model from file and onto device
def load_fcn_resnet101( n_classes, model_path=None ):
    # Load pretrained resnet model
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True,
        progress=True)
    
    # Reconfigure the pretrained model to different number of classes
    model = ReconfigureClassifier(model=model, n_classes=n_classes)
    
    # If model path is given, load statedict from file
    if model_path:
        # Load saved model
        model.load_state_dict(torch.load(model_path))
        
    # Return loaded model
    return model
