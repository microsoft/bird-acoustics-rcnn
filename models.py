# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils

import numpy as np
import math

from typing import List, Tuple
from scipy.special import legendre
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay

# VGG pytorch model is taken from:
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html

cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
class VGG16_pool(nn.Module):
    def __init__(self, cfg=cfg_vgg16, num_classes=10, init_weights=True):
        super(VGG16_pool, self).__init__()

        self.convBlock = self.make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.Dense1 = nn.Linear(512*7*7, 4096)
        self.Dense2 = nn.Linear(4096, 4096)
        self.Dense3 = nn.Linear(4096, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convBlock(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.Dense1(x))
        x = self.dropout1(x)
        x = F.relu(self.Dense2(x))
        x = self.Dense3(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Resnet pytorch model is taken from:
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                   **kwargs)

def resnet18(**kwargs):
    return ResNet(Bottleneck, [2, 2, 2, 2],
                   **kwargs)


class cnn(nn.Module):
    def __init__(self, cfg, init_weights=True):
        super(cnn, self).__init__()
        
        self.convBlock = self.make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((2,1))
        
        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.convBlock(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def make_layers(self, cfg):
        layers = []
        in_channels = 1
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)
    

class cnn_ts(nn.Module):
    def __init__(self, cfg_ts, init_weights=True):
        super(cnn_ts, self).__init__()
        
        self.conv = self.make_layers(cfg_ts)
        self.avgpool = nn.AdaptiveAvgPool2d((1,4))
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def make_layers(self, cfg):
        layers = []
        in_channels = 1
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,3), stride=(2,3))]
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)

class cnn_cnn(nn.Module):
    def __init__(self, 
                 cfg, 
                 cfg_ts, 
                 n_units_ts, 
                 num_classes=100):
        super(cnn_cnn, self).__init__()
        
        self.cnn = cnn(cfg)
        self.cnn1 = cnn_ts(cfg_ts)
        
        self.linear1 = nn.Linear(n_units_ts, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)
        
    def forward(self, x, hidden_state=None):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        x = c_out.view(batch_size, timesteps, -1)
        x = x.unsqueeze(1)
        x = self.cnn1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 1)
    
    
class cnn_lstm(nn.Module):
    def __init__(self, n_units, cfg, num_layers=2, num_classes=100):
        super(cnn_lstm, self).__init__()
        
        self.cnn = cnn(cfg)
        self.hidden_size = 512
        self.num_layers = num_layers
        self.rnn1 = nn.LSTM(input_size=n_units, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.linear1 = nn.Linear(self.hidden_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)
        
    def forward(self, x, hidden_state=None):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        x, _ = self.rnn1(r_in, hidden_state)
        x = x.sum(dim=1)
        x = self.linear1(x)
        # x = self.linear1(x[:, -1, :]) # feeding last ouput of seq to linear layer (OPTIONAL)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    
def Legendre(shape):
    if len(shape) != 2:
        raise ValueError("Legendre initializer assumes shape is 2D; "
                         "but shape=%s" % (shape,))
    return np.asarray([legendre(i)(np.linspace(-1, 1, shape[1]))
                       for i in range(shape[0])])
    

# LMU cell taken from: https://github.com/nengo/keras-lmu
# and converted to pytorch
class LMUCell(nn.Module):
    def __init__(self,
                 input_dim,
                 units,
                 order,
                 theta,  # relative to dt=1
                 method='zoh',
                 realizer=Identity(),    # TODO: Deprecate?
                 factory=LegendreDelay,  # TODO: Deprecate?
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer='lecun_uniform',
                 input_encoders_initial_val = 0,
                 hidden_encoders_initializer='lecun_uniform',
                 hidden_encoders_initial_val = 0,
                 memory_encoders_initializer='Constant',  # 'lecun_uniform',
                 memory_encoders_initial_val = 0,
                 input_kernel_initializer='glorot_normal',
                 input_kernel_initial_val = 0,
                 hidden_kernel_initializer='glorot_normal',
                 hidden_kernel_initial_val = 0,
                 memory_kernel_initializer='glorot_normal',
                 memory_kernel_initial_val = 0,
                 hidden_activation='tanh',
                 **kwargs):
        super(LMUCell,self).__init__()

        self.units = units
        self.order = order
        self.theta = theta
        self.method = method
        self.realizer = realizer
        self.factory = factory
        self.trainable_input_encoders = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel = trainable_input_kernel
        self.trainable_hidden_kernel = trainable_hidden_kernel
        self.trainable_memory_kernel = trainable_memory_kernel
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        self.hidden_activation = hidden_activation

        self._realizer_result = realizer(
            factory(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1., method=method)
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.state_size = (self.units, self.order)
        self.output_size = self.units

        def weight_mod(input_dim, output_dim, initialization,
                            constant_val = 0):

            w = torch.FloatTensor(input_dim, output_dim)
            w.requires_grad = True

            if initialization == 'lecun_uniform':
                torch.nn.init.kaiming_uniform_(w)
            elif initialization == 'glorot_normal':
                torch.nn.init.xavier_normal_(w)
            elif initialization == 'Constant':
                if np.size(constant_val) == 1:
                    torch.nn.init.constant_(w, constant_val)
                else:
                    w.data = torch.from_numpy(constant_val).float()
            elif initialization == 'Legendre':
                w.data = torch.from_numpy(Legendre((input_dim, output_dim))).float()
                
            elif initialization == 'uniform':
                stdv = 1.0 / math.sqrt(self.state_size[0])
                torch.nn.init.uniform_(w, -stdv, stdv)

            return w

        self.input_encoders = nn.Parameter(weight_mod(input_dim, 1, 
                                            initialization=input_encoders_initializer,                                     
                                            constant_val = input_encoders_initial_val)
                                          )
        if not self.trainable_input_encoders:
            self.input_encoders.requires_grad = False
        
        self.hidden_encoders = nn.Parameter(weight_mod(self.units, 1, 
                                            initialization=hidden_encoders_initializer,  
                                            constant_val = hidden_encoders_initial_val)
                                           )

        if not self.trainable_hidden_encoders:
            self.hidden_encoders.requires_grad = False
        
        self.memory_encoders = nn.Parameter(weight_mod(self.order, 1, 
                                            initialization='Constant',
                                            constant_val=0)
                                           )

        if not self.trainable_memory_encoders:
            self.memory_encoders.requires_grad = False
        
        self.input_kernel = nn.Parameter(weight_mod(input_dim, self.units, 
                                            initialization=input_kernel_initializer,
                                            constant_val = input_kernel_initial_val)
                                        )

        if not self.trainable_input_kernel:
            self.input_kernel.requires_grad = False
        
        self.hidden_kernel = nn.Parameter(weight_mod(self.units, self.units, 
                                            initialization=hidden_kernel_initializer, 
                                            constant_val = hidden_kernel_initial_val)
                                         )

        if not self.trainable_hidden_kernel:
            self.hidden_kernel.requires_grad = False
        
        self.memory_kernel = nn.Parameter(weight_mod(self.order, self.units, 
                                            initialization=memory_kernel_initializer, 
                                            constant_val = memory_kernel_initial_val)
                                         )

        if not self.trainable_memory_kernel:
            self.memory_kernel.requires_grad = False
        
        self.AT = nn.Parameter(weight_mod(self.order, self.order, 
                                initialization='Constant',
                                constant_val=self._A.T)  # transposed
                              )

        if not self.trainable_A:
            self.AT.requires_grad = False
        
        self.BT = nn.Parameter(weight_mod(1, self.order, 
                                initialization='Constant',
                                constant_val=self._B.T) # transposed
                              )
        if not self.trainable_B:
            self.BT.requires_grad = False
        

    def forward(self, inputs, states):
        h, m = states
        
        u = torch.mm(inputs, self.input_encoders) \
             + torch.mm(h, self.hidden_encoders) \
             + torch.mm(m, self.memory_encoders)
        
        m = m + torch.mm(m, self.AT) + torch.mm(u, self.BT)

        if self.hidden_activation == 'tanh':
            h = torch.tanh(
                torch.mm(inputs, self.input_kernel) +
                torch.mm(h, self.hidden_kernel) +
                torch.mm(m, self.memory_kernel)
                )
        elif self.hidden_activation == 'linear':
            h = torch.mm(inputs, self.input_kernel) \
                + torch.mm(h, self.hidden_kernel) \
                + torch.mm(m, self.memory_kernel)
            
        return h, (h, m)
    
    
# using https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# for custom LSTMs
class LMU(nn.Module):
    def __init__(self, inp_size = 1, order = 100, theta=100, output_dims = 1):
        super(LMU, self).__init__()
        
        self.units = output_dims
        self.order = order
        self.output_dims = output_dims
        self.lmu_cell = LMUCell(
                input_dim = inp_size,
                units=output_dims,
                order=order,
                theta = theta,
                input_encoders_initializer='uniform',
                hidden_encoders_initializer='uniform',
                memory_encoders_initializer='uniform',  # 'lecun_uniform',
                input_kernel_initializer='uniform',
                hidden_kernel_initializer='uniform',
                memory_kernel_initializer='uniform',
            )
    
    def forward(self, x, state):
        x = x.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(x)):
            out, state = self.lmu_cell(x[i], state)
            outputs += [out]
        
        return torch.stack(outputs).permute(1, 0, 2), state  # axes permuted to make output of shape B, seq_len, num_outputs
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.units),
                torch.zeros(batch_size, self.order)) 
    
# using https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# for custom LSTMs
class LMUGate(nn.Module):
    def __init__(self, inp_size = 1, order = 100, theta=100, output_dims = 1):
        super(LMUGate, self).__init__()
        
        self.units = output_dims
        self.order = order
        self.output_dims = output_dims
        self.lmu_cell = LMUCellGate(
                input_dim = inp_size,
                units=output_dims,
                order=order,
                theta = theta,
            )
    
    def forward(self, x, state):
        x = x.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(x)):
            out, state = self.lmu_cell(x[i], state)
            outputs += [out]
        
        return torch.stack(outputs).permute(1, 0, 2), state  # axes permuted to make output of shape B, seq_len, num_outputs
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.units),
                torch.zeros(batch_size, self.order)) 
    
    
class cnn_rnn(nn.Module):
    def __init__(self, 
                theta=500, 
                num_classes=100, 
                order=100,
                hidden_size = 512,
                cnnConfig = cfg_vgg16,
                rnnConfig = {
                    'LMU_0':
                        {'input_size':1024, 'h_states_ctr':1}
                }
                ):
        super(cnn_rnn, self).__init__()
        
        #         sample: GRU -> LMU
        #         rnnConfig = {
        #                     'GRU_0':
        #                         {'input_size':n_units, 'h_states_ctr':1}, 
        #                     'LMU_1':
        #                         {'input_size':self.hidden_size, 'h_states_ctr':2}, 
        #                     }
        
        self.cnn = cnn(cnnConfig)
        self.hidden_size = hidden_size
        self.order = order
        self.theta = theta
        self.rnnConfig = rnnConfig
        self.rnnBlock = self.make_rnn()
        self.linear1 = nn.Linear(self.hidden_size, 512) 
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)
        
    def forward(self, x, hidden_state):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in) 
        x = c_out.view(batch_size, timesteps, -1)
        
        ctr = 0
        for (rnn_name, config_), rnn in zip(self.rnnConfig.items(), self.rnnBlock):
            if config_['h_states_ctr']==1:
                h_s = hidden_state[ctr]
            else:
                h_s = tuple([hidden_state[ctr+i] for i in range(config_['h_states_ctr'])])
            ctr += config_['h_states_ctr']
            x, _ = rnn(x, h_s)
        
        x = x.sum(dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
    
    def make_rnn(self):
        layers = []
        for rnn_name, config_ in self.rnnConfig.items():
            if rnn_name.split('_')[0] == 'LMU':
                layers += [LMU(inp_size=config_['input_size'], 
                               order=self.order, theta=self.theta, 
                               output_dims=self.hidden_size)]
            elif rnn_name.split('_')[0] == 'GRU':
                layers += [nn.GRU(input_size=config_['input_size'], 
                                  hidden_size=self.hidden_size, batch_first=True)]
            elif rnn_name.split('_')[0] == 'LSTM':
                layers += [nn.LSTM(input_size=config_['input_size'], 
                                  hidden_size=self.hidden_size, batch_first=True)]
                
        return nn.ModuleList(layers)
        
    
    def init_hidden(self, batch_size):
        h_s = []
        for rnn_name in self.rnnConfig.keys():
            if rnn_name.split('_')[0] == 'LMU':
                h_s += [torch.zeros(batch_size, self.hidden_size)]
                h_s += [torch.zeros(batch_size, self.order)]
            elif rnn_name.split('_')[0] == 'GRU':
                h_s += [torch.zeros(1, batch_size, self.hidden_size)]
            elif rnn_name.split('_')[0] == 'LSTM':
                h_s += [torch.zeros(1, batch_size, self.hidden_size)]
                h_s += [torch.zeros(1, batch_size, self.hidden_size)]
        return tuple(h_s)