import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.utils import misc

from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
import warnings
import numpy as np

bn_axis = 1

"""(convolution => [BN] => ReLU/sigmoid)"""
class Conv_Bn_Relu(nn.Module):
    def __init__(self, in_channels, out_channels=32, 
        kernelSize=(3,3), strds=(1,1),
        useBias=False, dilatationRate=(1,1), 
        actv='relu', doBatchNorm=True
    ):

        super().__init__()
        if isinstance(kernelSize, int):
            kernelSize = (kernelSize, kernelSize)
        if isinstance(strds, int):
            strds = (strds, strds)

        self.conv_bn_relu = self.get_block(in_channels, out_channels, kernelSize,
            strds, useBias, dilatationRate, actv, doBatchNorm
        )

    def forward(self, input):
        return self.conv_bn_relu(input)


    def get_block(self, in_channels, out_channels, 
        kernelSize, strds,
        useBias, dilatationRate, 
        actv, doBatchNorm
    ):

        layers = []

        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernelSize, 
                stride=strds, dilation=dilatationRate, bias=useBias, padding='same', padding_mode='zeros'
            )

        torch.nn.init.xavier_uniform_(conv1.weight)

        layers.append(conv1)

        if doBatchNorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels,eps=1.001e-5))

        if actv == 'relu':
            layers.append(nn.ReLU())
        elif actv == 'sigmoid':
            layers.append(nn.Sigmoid())

        block = nn.Sequential(*layers)
        return block


"""Multiscale Conv Block"""
class Multiscale_Conv_Block(nn.Module):


    def __init__(self, in_channels, kernelSizes, 
        dilatationRates, out_channels=32, strds=(1,1),
        actv='relu', useBias=False
    ):

        super().__init__()

        self.conv_block_1 = Conv_Bn_Relu(in_channels=in_channels, out_channels=out_channels, kernelSize=kernelSizes[0],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[0], dilatationRates[0]))
            
        self.conv_block_2 = Conv_Bn_Relu(in_channels=in_channels, out_channels=out_channels, kernelSize=kernelSizes[1],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[1], dilatationRates[1]))

        self.conv_block_3 = Conv_Bn_Relu(in_channels=in_channels, out_channels=out_channels, kernelSize=kernelSizes[2],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[2], dilatationRates[2]))

        self.conv_block_4 = Conv_Bn_Relu(in_channels=in_channels, out_channels=out_channels, kernelSize=kernelSizes[3],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[3], dilatationRates[3]))


    def forward(self, input_map):

        conv0 = input_map

        conv1 = self.conv_block_1(conv0)
        conv2 = self.conv_block_2(conv0)
        conv3 = self.conv_block_3(conv0)
        conv4 = self.conv_block_4(conv0)

        output_map = torch.cat([conv1, conv2, conv3, conv4], dim=bn_axis)

        return output_map


"""Residual_Conv"""
class Residual_Conv(nn.Module):


    def __init__(self, in_channels, out_channels=32, 
        kernelSize=(3,3), strds=(1,1), actv='relu', 
        useBias=False, dilatationRate=(1,1)
    ):
        super().__init__()



        self.conv_block_1 = Conv_Bn_Relu(in_channels, out_channels, kernelSize=kernelSize, strds=strds, 
            actv='None', useBias=useBias, dilatationRate=dilatationRate, doBatchNorm=True
        )
        self.conv_block_2 = Conv_Bn_Relu(out_channels, out_channels, kernelSize=kernelSize, strds=strds, 
            actv='None', useBias=useBias, dilatationRate=dilatationRate, doBatchNorm=True
        )

        if actv == 'relu':
            self.activation = nn.ReLU()
        elif actv == 'sigmoid':
            self.activation = nn.Sigmoid()


    def forward(self, input):
        conv1 = self.conv_block_1(input)
        conv2 = self.conv_block_2(conv1)

        out = torch.add(conv1, conv2)
        out = self.activation(out)
        return out



class NuClick_NN(ModelABC):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net_name = 'NuClick'

        self.n_channels = n_channels
        self.n_classes = n_classes

        #-------------Conv_Bn_Relu blocks------------
        self.conv_block_1 = nn.Sequential(
            Conv_Bn_Relu(in_channels=self.n_channels, out_channels=64, kernelSize=7),
            Conv_Bn_Relu(in_channels=64, out_channels=32, kernelSize=5),
            Conv_Bn_Relu(in_channels=32, out_channels=32, kernelSize=3)
        )

        self.conv_block_2 = nn.Sequential(
            Conv_Bn_Relu(in_channels=64, out_channels=64),
            Conv_Bn_Relu(in_channels=64, out_channels=32),
            Conv_Bn_Relu(in_channels=32, out_channels=32)
        )

        self.conv_block_3 = Conv_Bn_Relu(in_channels=32, out_channels=self.n_classes,
            kernelSize=(1,1), actv=None, useBias=True, doBatchNorm=False)

        #-------------Residual_Conv blocks------------
        self.residual_block_1 = nn.Sequential(
            Residual_Conv(in_channels=32, out_channels=64),
            Residual_Conv(in_channels=64, out_channels=64)
        )

        self.residual_block_2 = Residual_Conv(in_channels=64, out_channels=128)

        self.residual_block_3 = Residual_Conv(in_channels=128, out_channels=128)

        self.residual_block_4 = nn.Sequential(
            Residual_Conv(in_channels=128, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256)
        )

        self.residual_block_5 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512)
        )

        self.residual_block_6 = nn.Sequential(
            Residual_Conv(in_channels=512, out_channels=1024),
            Residual_Conv(in_channels=1024, out_channels=1024)
        )

        self.residual_block_7 = nn.Sequential(
            Residual_Conv(in_channels=1024, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=256)
        )

        self.residual_block_8 = Residual_Conv(in_channels=512, out_channels=256)

        self.residual_block_9 = Residual_Conv(in_channels=256, out_channels=256)

        self.residual_block_10 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=128),
            Residual_Conv(in_channels=128, out_channels=128)
        )

        self.residual_block_11 = Residual_Conv(in_channels=128, out_channels=64)

        self.residual_block_12 = Residual_Conv(in_channels=64, out_channels=64)


        #-------------Multiscale_Conv_Block blocks------------
        self.multiscale_block_1 = Multiscale_Conv_Block(in_channels=128, out_channels=32,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,3,6]
        )

        self.multiscale_block_2 = Multiscale_Conv_Block(in_channels=256, out_channels=64,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,2,3]
        )

        self.multiscale_block_3 = Multiscale_Conv_Block(in_channels=64, out_channels=16,
            kernelSizes=[3,3,5,7], dilatationRates=[1,3,2,6]
        )
            
        #-------------MaxPool2d blocks------------
        self.pool_block_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_3 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_4 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_5 = nn.MaxPool2d(kernel_size=(2,2))

        #-------------ConvTranspose2d blocks------------
        self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_5 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=2, stride=(2,2),
        )

    def forward(self, input):

        conv1 = self.conv_block_1(input)    
        pool1 = self.pool_block_1(conv1)     

        conv2 = self.residual_block_1(pool1) 
        pool2 = self.pool_block_2(conv2)    

        conv3 = self.residual_block_2(pool2)
        conv3 = self.multiscale_block_1(conv3)  
        conv3 = self.residual_block_3(conv3)    
        pool3 = self.pool_block_3(conv3)    

        conv4 = self.residual_block_4(pool3)    
        pool4 = self.pool_block_4(conv4)    

        conv5 = self.residual_block_5(pool4) 
        pool5 = self.pool_block_5(conv5)    

        conv51 = self.residual_block_6(pool5) 

        up61 = torch.cat([self.conv_transpose_1(conv51),conv5], dim=1)  
        conv61 = self.residual_block_7(up61)    
        
        up6 = torch.cat([self.conv_transpose_2(conv61), conv4], dim=1)  
        conv6 = self.residual_block_8(up6) 
        conv6 = self.multiscale_block_2(conv6)  
        conv6 = self.residual_block_9(conv6)    

        up7 = torch.cat([self.conv_transpose_3(conv6), conv3], dim=1)   
        conv7 = self.residual_block_10(up7)     

        up8 = torch.cat([self.conv_transpose_4(conv7), conv2], dim=1)   
        conv8 = self.residual_block_11(up8)     
        conv8 = self.multiscale_block_3(conv8)  
        conv8 = self.residual_block_12(conv8)   

        up9 = torch.cat([self.conv_transpose_5(conv8), conv1], dim=1)   
        conv9 = self.conv_block_2(up9)  

        conv10 = self.conv_block_3(conv9)   
        
        return conv10



    
    @staticmethod
    def postproc(preds, thresh=0.33, minSize=10, minHole=30, doReconstruction=False, nucPoints=None):
        masks = preds > thresh
        masks = remove_small_objects(masks, min_size=minSize)
        masks = remove_small_holes(masks, area_threshold=minHole)
        if doReconstruction:
            for i in range(len(masks)):
                thisMask = masks[i]
                thisMarker = nucPoints[i, 0, :, :] > 0
                
                try:
                    thisMask = reconstruction(thisMarker, thisMask, footprint=disk(1))
                    masks[i] = np.array([thisMask])
                except:
                    warnings.warn('Nuclei reconstruction error #' + str(i))
        return masks   


    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        model.eval()
        device = misc.select_device(on_gpu)

        #Assume batch_data is NCHW
        imgs_points = batch_data
        imgs_points_device = imgs_points.to(device).type(torch.float32)

        with torch.inference_mode():
            output = model(imgs_points_device)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)

        return output.cpu().numpy()