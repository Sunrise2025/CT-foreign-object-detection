# Code adapted from here:
# https://github.com/joe3141592/PyTorch-CAM
import torch
import matplotlib.pyplot as plt
import skimage.transform

from aug_accuracy import InputError


class ActivationMap:
    def __init__(self, model):
        '''The initializer needs to find the final layer before avgpool and insert a hook to extract the last feature map to self.conv_features
        
        '''
        if model.nn_type == "resnet50":
            # The last layer before Linear in ResNet
            final_layer = model.net[1].layer4
            # Weights of fc
            self.final_weights = list(model.net[1].parameters())[-2]
        elif model.nn_type == "efficientnetb4":
            final_layer = model.net[1].features[8]
            self.final_weights = list(model.net[1].parameters())[-2]
        else:
            raise InputError("Unknown sample type. Got {}".format(model.nn_type))
        
        final_layer.register_forward_hook(self.__hook)
        self.conv_features = None
        self.model = model
        
    def __hook(self, _, inp, out):
        '''ResNet 50 case:
        This hook will output a feature map with a shape of [1, 2048, 12, 15]
        Every 12x15 image will be compressed to a single value by adaptive average pooling layer avgpool
        2048 channels will be used as an input of the last Linear layer fc
        Batch size is assumed to be 1 later
        '''
        self.conv_features = out
        
    def generateCAM(self, out_class):
        '''ResNet 50 case:
        Every feature map increase influences class probabilities according to the final_weights - weights of fc.
        Matrix multiplication is used to move from 2048 feature maps to 3 class activation maps.
        '''
        # Convert 12x15 matrix into 1x180 vector, assumes that batch_size = 1
        batch_size, n_kernels, h, w = self.conv_features.size()
        assert batch_size == 1
        flat_features = self.conv_features.view(n_kernels, h * w)
        
        # Use weights of fc Linear layer to go from 2048 channels to 3 output classes. Then return 12x15 shape.
        maps = self.final_weights.mm(flat_features)
        out_classes = self.final_weights.size()[0]
        maps = maps.view(out_classes, h, w)
        
        # Check that the map is requested for a valid class and normalize it
        assert out_class < out_classes
        cam = maps[out_class]
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        cam = (cam * 255).int()
        cam = cam.cpu().detach().numpy()
        
        return cam
    
    def visualize(self, input_image, tg, out_class, img_fname):
        overlay = self.generateCAM(out_class)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                
        for ax in (ax1, ax2):
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            
        ax1.set_title("Input image", fontsize=16)
        ax1.imshow(input_image[0,0,:], cmap='Greys')
        ax1.text(20., 20., 'GT    - {}'.format(tg), fontsize=16)
        ax1.text(20., 50., 'Pred - {}'.format(out_class), fontsize=16)
            
        ax2.set_title("Activation", fontsize=16)
        ax2.imshow(input_image[0,0,:], cmap='Greys')
        
        ax2.imshow(skimage.transform.resize(overlay, input_image[0,0,:].shape), alpha=0.5, cmap='jet')
                                
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
                
        fig.savefig(img_fname)
        plt.close()
        
    
            
