import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import math
from .gumbel_softmax import GumbleSoftmax



class dynamic_fusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super(dynamic_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b,c)
        attention = self.fc(y)
        return attention.view(b,c,1,1)
    
    def forward_calc_flops(self, x):
        b, c, h, w = x.size()
        flops = c*h*w
        y = self.avg_pool(x).view(b,c)
        attention = self.fc(y)
        flops += c*c//self.reduction*2 + c
        return attention.view(b,c,1,1), flops


class TSM(nn.Module):
    def __init__(self):
        super(TSM, self).__init__()
        self.fold_div = 8

    def forward(self, x, n_segment):
        x = self.shift(x, n_segment, fold_div=self.fold_div)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3):
        if type(n_segment) is int:
            nt, c, h, w = x.size()
            n_batch = nt // n_segment
            x = x.view(n_batch, n_segment, c, h, w)

            fold = c // fold_div
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            shift_out = out.view(nt, c, h, w)
        else:
            num_segment = int(n_segment.sum())
            ls = n_segment
            bool_list = ls > 0
            bool_list = bool_list.view(-1)

            shift_out = torch.zeros_like(x)
            x = x[bool_list]
            nt, c, h, w = x.size()
            x = x.view(-1, num_segment, c, h, w)

            fold = c // fold_div
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            out = out.view(-1, c, h, w)
            shift_out[bool_list] = out

        return shift_out


class Bottleneck_ample(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, last_relu=True, patch_groups=1, 
                 base_scale=2, is_first=False):
        super(Bottleneck_ample, self).__init__()
        self.num_segments = num_segments
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=1)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.tsm = TSM()
       
        self.downsample = downsample
        self.have_pool = False
        self.have_1x1conv2d = False

        self.first_downsample = nn.AvgPool2d(3, stride=2, padding=1) if (base_scale == 4 and is_first) else None
            
        if self.downsample is not None:
            self.have_pool = True
            if len(self.downsample) > 1:
                self.have_1x1conv2d = True
        
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x, list_little, activate_tsm=False):
        if self.first_downsample is not None:
            x = self.first_downsample(x)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if activate_tsm:
            out = self.tsm(x, list_little)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out

    def forward_calc_flops(self, x, list_little, activate_tsm=False):
        flops = 0
        if self.first_downsample is not None:
            x = self.first_downsample(x)
            _, c, h, w = x.shape
            flops += 9 * c * h * w

        residual = x        
        if self.downsample is not None:
            c_in = x.shape[1]
            residual = self.downsample(x)
            _, c, h, w = residual.shape
            if self.have_pool:
                flops += 9 * c_in * h * w
            if self.have_1x1conv2d:
                flops += c_in * c * h * w
        
        if activate_tsm:
            out = self.tsm(x, list_little)
        else:
            out = x

        c_in = out.shape[1]
        out = self.conv1(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w  / self.conv1.groups

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w * 9 / self.conv2.groups
        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        _,c_out,h,w = out.shape
        flops += c_in * c_out * h * w / self.conv3.groups
        out = self.bn3(out)
        
        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out, flops

class Bottleneck_focal(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, last_relu=True, patch_groups=1, base_scale=2, is_first = True):
        super(Bottleneck_focal, self).__init__()
        self.num_segments = num_segments
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False, groups=patch_groups)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=patch_groups)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False, groups=patch_groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.tsm = TSM()
        self.downsample = downsample

        self.stride = stride
        self.last_relu = last_relu
        self.patch_groups = patch_groups

    def forward(self, x, mask, activate_tsm=False):
        residual = x
        if self.downsample is not None:     # skip connection before mask
            residual = self.downsample(x)

        if activate_tsm:
            out = self.tsm(x, self.num_segments)
        else:
            out = x
        out = out * mask

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out


    def forward_calc_flops(self, x, mask, activate_tsm=False):
        residual = x
        flops = 0
        if self.downsample is not None:     # skip connection before mask
            c_in = x.shape[1]
            residual = self.downsample(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3]
        
        if activate_tsm:
            out = self.tsm(x, self.num_segments)
        else:
            out = x
        out = out * mask
        select_ratio = torch.mean(mask)
           
        c_in = out.shape[1]
        out = self.conv1(out)
        _,c_out,h,w = out.shape
        flops += select_ratio * c_in * c_out * h * w  / self.conv1.groups

        out = self.bn1(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv2(out)
        _,c_out,h,w = out.shape
        flops += select_ratio * c_in * c_out * h * w * 9 / self.conv2.groups
        out = self.bn2(out)
        out = self.relu(out)

        c_in = c_out
        out = self.conv3(out)
        _,c_out,h,w = out.shape
        flops += select_ratio * c_in * c_out * h * w / self.conv3.groups
        out = self.bn3(out)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out, flops



class navigation(nn.Module):
    def __init__(self, inplanes=64, num_segments=8):
        super(navigation,self).__init__()
        self.num_segments = num_segments
        self.conv_pool = nn.Conv2d(inplanes, 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_gs = nn.Conv2d(2*num_segments, 2*num_segments, kernel_size=1, padding=0, stride=1, bias=True, groups=num_segments)
        self.conv_gs.bias.data[:2*num_segments:2] = 1.0
        self.conv_gs.bias.data[1:2*num_segments+1:2] = 10.0
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1.0): 
        gates = self.pool(x)
        gates = self.conv_pool(gates)
        gates = self.bn(gates)
        gates = self.relu(gates)

        batch = x.shape[0] // self.num_segments

        gates = gates.view(batch, self.num_segments*2,1,1)
        gates = self.conv_gs(gates)

        gates = gates.view(batch, self.num_segments, 2, 1, 1)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        mask = gates[:, :, 1, :, :]
        mask = mask.view(x.shape[0],1,1,1)

        return mask

    def forward_calc_flops(self, x, temperature=1.0):
        flops = 0

        flops += x.shape[1] * x.shape[2] * x.shape[3]
        gates = self.pool(x)

        c_in = gates.shape[1]
        gates = self.conv_pool(gates)
        flops += c_in * gates.shape[1] * gates.shape[2] * gates.shape[3]
        gates = self.bn(gates)
        gates = self.relu(gates)

        batch = x.shape[0] // self.num_segments

        gates = gates.view(batch, self.num_segments*2,1,1)
        gates = self.conv_gs(gates)
        flops += self.num_segments * 2 * gates.shape[1] * gates.shape[2] * gates.shape[3] / self.conv_gs.groups

        gates = gates.view(batch, self.num_segments, 2, 1, 1)
        gates = self.gs(gates, temp=temperature, force_hard=True)
        mask = gates[:, :, 1, :, :]
        mask = mask.view(x.shape[0],1,1,1)
        
        return mask, flops


class AFModule(nn.Module):
    def __init__(self, block_ample, block_focal, in_channels, out_channels, blocks, stride, patch_groups, alpha=1, num_segments=8):
        super(AFModule, self).__init__()
        self.num_segments = num_segments
        self.patch_groups = patch_groups
        self.relu = nn.ReLU(inplace=True)

        frame_gen_list = []
        for i in range(blocks - 1):
            frame_gen_list.append(navigation(inplanes=int(out_channels // alpha),num_segments=num_segments)) if i!=0 else frame_gen_list.append(navigation(inplanes=in_channels,num_segments=num_segments))
        self.list_gen = nn.ModuleList(frame_gen_list)

        self.base_module = self._make_layer(block_ample, in_channels, int(out_channels // alpha), num_segments, blocks - 1, 2, last_relu=False)       
        self.refine_module = self._make_layer(block_focal, in_channels, out_channels, num_segments, blocks - 1, 1, last_relu=False)
        
        self.alpha = alpha
        if alpha != 1:
            self.base_transform = nn.Sequential(
                nn.Conv2d(int(out_channels // alpha), out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.att_gen = dynamic_fusion(channel=out_channels, reduction=16)
        self.fusion = self._make_layer(block_ample, out_channels, out_channels, num_segments, 1, stride=stride)

    def _make_layer(self, block, inplanes, planes, num_segments, blocks, stride=1, last_relu=True, base_scale=2):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if blocks == 1:         # fuse, is not the first of a base branch
            layers.append(block(inplanes, planes, num_segments, stride=stride, downsample=downsample,
                                patch_groups=self.patch_groups, base_scale=base_scale, is_first = False))
        else:
            layers.append(block(inplanes, planes, num_segments, stride, downsample,patch_groups=self.patch_groups, 
                             base_scale=base_scale, is_first = True))
            for i in range(1, blocks):
                layers.append(block(planes, planes, num_segments,
                                    last_relu=last_relu if i == blocks - 1 else True, 
                                    patch_groups=self.patch_groups, base_scale=base_scale, is_first = False))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1e-8, activate_tsm=False):
        b,c,h,w = x.size()
        x_big = x
        x_little = x
        _masks = []

        for i in range(len(self.base_module)):
            mask = self.list_gen[i](x_little, temperature=temperature)
            _masks.append(mask)

            x_little = self.base_module[i](x_little, self.num_segments, activate_tsm)
            x_big = self.refine_module[i](x_big, mask, activate_tsm)
        
        if self.alpha != 1:
            x_little = self.base_transform(x_little)

        _,_,h,w = x_big.shape
        x_little = F.interpolate(x_little, size = (h,w))
        att = self.att_gen(x_little+x_big)
        out = self.relu(att*x_little + (1-att)*x_big)
        out = self.fusion[0](out, self.num_segments, activate_tsm)
        return out, _masks

    def forward_calc_flops(self, x, temperature=1e-8, activate_tsm=False):
        flops = 0
        b,c,h,w = x.size()

        x_big = x
        x_little = x
        _masks = []
        
        for i in range(len(self.base_module)):
            mask, _flops = self.list_gen[i].forward_calc_flops(x_little, temperature=temperature)
            _masks.append(mask)
            flops += _flops * b

            x_little, _flops = self.base_module[i].forward_calc_flops(x_little, self.num_segments, activate_tsm)
            flops += _flops * b
            x_big, _flops = self.refine_module[i].forward_calc_flops(x_big, mask, activate_tsm)
            flops += _flops * b

        c = x_little.shape[1]
        _,_, h,w = x_big.shape
        if self.alpha != 1:
            x_little = self.base_transform(x_little)
            flops += b * c * x_little.shape[1] * x_little.shape[2] * x_little.shape[3]
        
        x_little = F.interpolate(x_little, size = (h,w))
        att, _flops = self.att_gen.forward_calc_flops(x_little+x_big)
        flops += _flops * b
        out = self.relu(att*x_little + (1-att)*x_big)
        out, _flops = self.fusion[0].forward_calc_flops(out, self.num_segments, activate_tsm)
        flops += _flops * b

        seg = b / self.num_segments
        flops = flops / seg

        return out, _masks, flops

class AFResNet(nn.Module):
    def __init__(self, block_ample, block_focal, layers, width=1.0, patch_groups=1, alpha=1, shift=True, num_segments=8, num_classes=1000):
        num_channels = [int(64*width), int(128*width), int(256*width), 512]

        self.num_segments = num_segments
        self.activate_tsm = shift
        self.inplanes = 64
        super(AFResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = AFModule(block_ample, block_focal, num_channels[0], num_channels[0]*block_ample.expansion, 
                               layers[0], stride=2, patch_groups=patch_groups, alpha=alpha, num_segments=num_segments)
        self.layer2 = AFModule(block_ample, block_focal, num_channels[0]*block_ample.expansion,
                               num_channels[1]*block_ample.expansion, layers[1], stride=2, patch_groups=patch_groups, alpha=alpha, num_segments=num_segments)
        self.layer3 = AFModule(block_ample, block_focal, num_channels[1]*block_ample.expansion,
                               num_channels[2]*block_ample.expansion, layers[2], stride=1, patch_groups=patch_groups, alpha=alpha, num_segments=num_segments)
        self.layer4 = self._make_layer(num_segments,
            block_ample, num_channels[2]*block_ample.expansion, num_channels[3]*block_ample.expansion, layers[3], stride=2)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3]*block_ample.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if 'gs' in str(k):
                #     m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck_ample):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, num_segments, block, inplanes, planes, blocks, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, num_segments, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, num_segments))

        return nn.ModuleList(layers)

    def forward(self, x, temperature=1.0):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        _masks = []
        x1, mask = self.layer1(x, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        x2, mask = self.layer2(x1, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        x3, mask = self.layer3(x2, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        x4 = x3
        for i in range(len(self.layer4)):
            x4 = self.layer4[i](x4, self.num_segments, self.activate_tsm)

        x = self.gappool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, _masks

    def forward_calc_flops(self, x, temperature=1.0):
        flops = 0
        c_in = x.shape[1]
        x = self.conv1(x)
        flops += self.num_segments * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        flops += self.num_segments * x.numel() / x.shape[0] * 9

        _masks = []
        x1, mask, _flops = self.layer1.forward_calc_flops(x, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        flops += _flops
        x2, mask, _flops = self.layer2.forward_calc_flops(x1, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        flops += _flops
        x3, mask, _flops = self.layer3.forward_calc_flops(x2, temperature=temperature, activate_tsm=self.activate_tsm)
        _masks.extend(mask)
        flops += _flops
        x4 = x3
        for i in range(len(self.layer4)):
            x4, _flops = self.layer4[i].forward_calc_flops(x4, self.num_segments, self.activate_tsm)
            flops += _flops * self.num_segments
        flops += self.num_segments * x4.shape[1] * x4.shape[2] * x4.shape[3]
        x = self.gappool(x4)
        x = x.view(x.size(0), -1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += self.num_segments * c_in * x.shape[1]

        return x, _masks, flops


def AF_resnet(depth, patch_groups=1, width=1.0, alpha=1, shift=False, num_segments=8, **kwargs):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
    }[depth]
    block = Bottleneck_ample
    block_focal = Bottleneck_focal
    model = AFResNet(block_ample=block, block_focal=block_focal, layers=layers, patch_groups=patch_groups, 
                     width=width, alpha=alpha, shift=shift, num_segments=num_segments, **kwargs)
    return model


def AF_resnet50(pretrained=False, path_backbone = '.../.../checkpoint/ImageNet/AF-ResNet50.pth.tar', shift=False, num_segments=8, **kwargs):    
    model = AF_resnet(depth=50, patch_groups=2, alpha=2, shift=shift, num_segments=num_segments, **kwargs)          
    if pretrained:
        checkpoint = torch.load(path_backbone, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k[7:] in new_state_dict):
                new_state_dict.update({k[7:]:v})
        model.load_state_dict(new_state_dict)
    return model


def AF_resnet101(pretrained=False, path_backbone = '.../.../checkpoint/ImageNet/AF-ResNet101.pth.tar', shift=False, num_segments=8, **kwargs):    
    model = AF_resnet(depth=101, patch_groups=2, alpha=2, shift=shift, num_segments=num_segments, **kwargs)          
    if pretrained:
        checkpoint = torch.load(path_backbone, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k[7:] in new_state_dict):
                new_state_dict.update({k[7:]:v})
        model.load_state_dict(new_state_dict)
    return model