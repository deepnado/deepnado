"""
Derived from file at https://github.com/mit-ll/tornet/blob/main/tornet/models/torch/cnn_baseline.py
"""

from typing import Dict, List, Tuple, Any
import numpy as np

import torch
from torch import nn

from deepnado.models.coordconv import CoordConv2D
from deepnado.common.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


class TornadoLikelihood(nn.Module):
    """
    Produces 2D tornado likelihood field
    """
    def __init__(self,  shape:Tuple[int]=(2,120,240),
                        c_shape:Tuple[int]=(2,120,240),
                        input_variables:List[str]=ALL_VARIABLES,
                        start_filters:int=64,
                        background_flag:float=-3.0,
                        include_range_folded:bool=True):
        super(TornadoLikelihood, self).__init__()
        self.input_shape=shape
        self.n_sweeps=shape[0]
        self.c_shape=c_shape
        self.input_variables=input_variables
        self.start_filters=start_filters
        self.background_flag=background_flag
        self.include_range_folded=include_range_folded
        
        # Set up normalizers
        self.input_norm_layers = {}
        for v in input_variables:
            min_max = np.array(CHANNEL_MIN_MAX[v]) # [2,]
            scale = 1/(min_max[1]-min_max[0])
            offset = min_max[0]
            self.input_norm_layers[v] = NormalizeVariable(scale,offset)
            
        # Processing blocks
        in_channels = (len(input_variables)+int(self.include_range_folded))*self.n_sweeps
        in_coords=self.c_shape[0]
        self.blk1 = VggBlock(in_channels,in_coords,start_filters,   kernel_size=3,  n_convs=2, drop_rate=0.1)   # (60,120)
        self.blk2 = VggBlock(start_filters,in_coords,2*start_filters, kernel_size=3,  n_convs=2, drop_rate=0.1)  # (30,60)
        self.blk3 = VggBlock(2*start_filters,in_coords,4*start_filters, kernel_size=3,  n_convs=3, drop_rate=0.1)  # (15,30)
        self.blk4 = VggBlock(4*start_filters,in_coords,8*start_filters, kernel_size=3,  n_convs=3, drop_rate=0.1)  # (7,15)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=8*start_filters, out_channels=512, kernel_size=(1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)),
             nn.ReLU()
        )
        self.conv_out = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,1))
        
        
    def _normalize_inputs(self,data):
        normed_data = {}
        for v in self.input_variables:
            normed_data[v] = self.input_norm_layers[v](data[v])
        return normed_data
    
    def forward(self,data: Dict[str,Any]):
        """
        Assumes data contains radar varialbes on [batch,tilt,az,rng]
        and coordinate tensor
        """
        # extract inputs
        data = data[0]
        x = {v:data[v] for v in self.input_variables} # each [batch,tilt,Az,Rng]
        c = data['coordinates']
        
        # normalize
        x = self._normalize_inputs(x) # each [batch,tilt,Az,Rng]
        
        # concatenate along channel (tilt) dim
        x = torch.cat([x[v] for v in self.input_variables],axis=1) #  [batch,tilt*len(input_variables)*2,Az,Rng]
        
        # Remove nan's
        x = torch.where(torch.isnan(x),self.background_flag,x)
        
        # concat range_Folded mask
        if self.include_range_folded:
            x = torch.cat((x,data['range_folded_mask']),axis=1)
        
        # process
        x,c=self.blk1((x,c))
        x,c=self.blk2((x,c))
        x,c=self.blk3((x,c))
        x,c=self.blk4((x,c))
        x = self.head(x)

        # output single channel heatmap for likelihood field
        x = self.conv_out(x)

        return x


class NormalizeVariable(nn.Module):
    """
    Normalize input tensor as (X-offset)*scale
    """
    def __init__(self, scale, offset):
        super(NormalizeVariable, self).__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.register_buffer('offset', torch.tensor(offset))

    def forward(self, x):
        return (x - self.offset) * self.scale


class VggBlock(nn.Module):
    """
    Processing block based on VGG19, with coord conv layers
    """
    def __init__(self, input_image_channels,
                       input_coordinate_channels,
                       n_output_channels,
                       kernel_size=3,
                       n_convs=3,
                       drop_rate=0.1):
        super(VggBlock, self).__init__()
        self.input_image_channels=input_image_channels
        self.input_coordinate_channels=input_coordinate_channels
        self.n_output_channels=n_output_channels
        self.kernel_size=kernel_size
        self.n_convs=n_convs
        self.drop_rate=drop_rate

        self.steps = []
        for k in range(n_convs):
            if k==0:
                in_channels=input_image_channels
            else:
                in_channels=n_output_channels
            self.steps.append(CoordConv2D(in_image_channels=in_channels,
                                          in_coord_channels=input_coordinate_channels,
                                          out_channels=n_output_channels, 
                                          kernel_size=kernel_size,
                                          padding='same',
                                          activation='relu'))
        self.conv_block = nn.Sequential(*self.steps)
        self.mx=nn.MaxPool2d(2, stride=2)
        self.mc=nn.MaxPool2d(2, stride=2)
        if drop_rate>0:
            self.drop=nn.Dropout(p=drop_rate)
        else:
            self.drop=None

    def forward(self, inputs):
        x,c=inputs
        x,c=self.conv_block((x,c))
        x=self.mx(x)
        c=self.mc(c)
        if self.drop:
            x=self.drop(x)
        return x,c