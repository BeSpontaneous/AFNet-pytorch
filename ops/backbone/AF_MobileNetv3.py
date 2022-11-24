import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from .gumbel_softmax import GumbleSoftmax


__all__ = ['AF_mobilenetv3']



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


class dynamic_fusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super(dynamic_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel),
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
    
    
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    def forward_calc_flops(self, x):
        b, c, h, w = x.size()
        flops = c*h*w
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        flops += c*c//self.reduction*2 + c
        return x * y, flops


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual_ample(nn.Module):
    def __init__(self, n_segment, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual_ample, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        self.tsm = TSM()
        self.inp = inp
        self.hidden_dim = hidden_dim
        self.use_se = use_se

        if inp == hidden_dim:
                # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # Squeeze-and-Excite
            self.se = SELayer(hidden_dim) if use_se else nn.Identity()
                # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
                # pw
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
                # Squeeze-and-Excite
            self.se = SELayer(hidden_dim) if use_se else nn.Identity()
            self.act2 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # pw-linear
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x, list_little):
        residual = x
        if self.inp == self.hidden_dim: 
            x = self.tsm(x, list_little)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.se(x)
            x = self.conv2(x)
            x = self.bn2(x)
        else:
            x = self.tsm(x, list_little)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.se(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.bn3(x)
        
        if self.identity:
            return x + residual
        else:
            return x

    def forward_calc_flops(self, x, list_little):
        flops = 0    
        residual = x
        if self.inp == self.hidden_dim:
            x = self.tsm(x, list_little)

            c_in = x.shape[1]
            x = self.conv1(x)
            flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] / self.conv1.groups
            
            x = self.bn1(x)
            x = self.act1(x)
            if self.use_se == True:
                x, _flops = self.se.forward_calc_flops(x)
                flops += _flops
            else:
                x = self.se(x)
            
            c_in = x.shape[1]
            x = self.conv2(x)
            flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv2.kernel_size[0] * self.conv2.kernel_size[1] / self.conv2.groups
            x = self.bn2(x)
        else:
            x = self.tsm(x, list_little)

            c_in = x.shape[1]
            x = self.conv1(x)
            flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] / self.conv1.groups
            x = self.bn1(x)
            x = self.act1(x)
            
            c_in = x.shape[1]
            x = self.conv2(x)
            flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv2.kernel_size[0] * self.conv2.kernel_size[1] / self.conv2.groups
            x = self.bn2(x)
            if self.use_se == True:
                x, _flops = self.se.forward_calc_flops(x)
                flops += _flops
            else:
                x = self.se(x)
            x = self.act2(x)
            
            c_in = x.shape[1]
            x = self.conv3(x)
            flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv3.kernel_size[0] * self.conv3.kernel_size[1] / self.conv3.groups
            x = self.bn3(x)
        if self.identity:
            return x + residual, flops
        else:
            return x, flops
        

class InvertedResidual_focal(nn.Module):
    def __init__(self, n_segment, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual_focal, self).__init__()
        assert stride in [1, 2]
        self.n_segment = n_segment
        self.identity = stride == 1 and inp == oup
        self.tsm = TSM()
        if stride != 1 or inp != oup:
            self.res_connect = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, bias=False, groups=2),
            nn.BatchNorm2d(oup)
            )
        
        self.inp = inp
        self.hidden_dim = hidden_dim
        self.use_se = use_se

        if inp == hidden_dim:
                # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # Squeeze-and-Excite
            self.se = SELayer(hidden_dim) if use_se else nn.Identity()
                # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=2, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
                # pw
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=2, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
                # Squeeze-and-Excite
            self.se = SELayer(hidden_dim) if use_se else nn.Identity()
            self.act2 = h_swish() if use_hs else nn.ReLU(inplace=True)
                # pw-linear
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=2, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x, list_big):
        if self.identity:
            residual = x
        else:
            residual = self.res_connect(x)
            
        if self.inp == self.hidden_dim:
            x = self.tsm(x, self.n_segment)
            x = x * list_big
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.se(x)
            x = self.conv2(x)
            x = self.bn2(x)
        else:
            x = self.tsm(x, self.n_segment)
            x = x * list_big
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.se(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.bn3(x)
        
        return x + residual


    def forward_calc_flops(self, x, list_big):
        flops = 0    
        
        if self.identity:
            residual = x
        else:
            c_in = x.shape[1]
            residual = self.res_connect(x)
            flops += c_in * residual.shape[1] * residual.shape[2] * residual.shape[3] / self.res_connect[0].groups
        
        if self.inp == self.hidden_dim:
            x = self.tsm(x, self.n_segment)
            x = x * list_big
            select_ratio = torch.mean(list_big)
            # select_ratio = 1

            c_in = x.shape[1]
            x = self.conv1(x)
            flops += select_ratio * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] / self.conv1.groups
            
            x = self.bn1(x)
            x = self.act1(x)
            if self.use_se == True:
                x, _flops = self.se.forward_calc_flops(x)
                flops += select_ratio * _flops
            else:
                x = self.se(x)
            
            c_in = x.shape[1]
            x = self.conv2(x)
            flops += select_ratio * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv2.kernel_size[0] * self.conv2.kernel_size[1] / self.conv2.groups
            x = self.bn2(x)
        else:
            x = self.tsm(x, self.n_segment)
            x = x * list_big
            select_ratio = torch.mean(list_big)
            # select_ratio = 1

            c_in = x.shape[1]
            x = self.conv1(x)
            flops += select_ratio * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] / self.conv1.groups
            x = self.bn1(x)
            x = self.act1(x)
            
            c_in = x.shape[1]
            x = self.conv2(x)
            flops += select_ratio * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv2.kernel_size[0] * self.conv2.kernel_size[1] / self.conv2.groups
            x = self.bn2(x)
            if self.use_se == True:
                x, _flops = self.se.forward_calc_flops(x)
                flops += select_ratio * _flops
            else:
                x = self.se(x)
            x = self.act2(x)
            
            c_in = x.shape[1]
            x = self.conv3(x)
            flops += select_ratio * c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv3.kernel_size[0] * self.conv3.kernel_size[1] / self.conv3.groups
            x = self.bn3(x)
            
        return x + residual, flops



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
        list_big = gates[:, :, 1, :, :]
        list_big = list_big.view(x.shape[0],1,1,1)

        return list_big

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
        list_big = gates[:, :, 1, :, :]
        list_big = list_big.view(x.shape[0],1,1,1)
        
        return list_big, flops



class AFMobileNetV3(nn.Module):
    def __init__(self, num_segments, num_class, cfgs_head, cfgs_stage1, cfgs_stage2_ample, 
                       cfgs_stage2_focal, cfgs_stage2_fuse, cfgs_stage3_ample, cfgs_stage3_focal, 
                       cfgs_stage3_fuse, cfgs_stage4, cfgs_stage5, mode, width_mult=1.):
        super(AFMobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.num_segments = num_segments
        self.cfgs_head = cfgs_head
        self.cfgs_stage1 = cfgs_stage1
        self.cfgs_stage2_ample = cfgs_stage2_ample
        self.cfgs_stage2_focal = cfgs_stage2_focal
        self.cfgs_stage2_fuse = cfgs_stage2_fuse
        self.cfgs_stage3_ample = cfgs_stage3_ample
        self.cfgs_stage3_focal = cfgs_stage3_focal
        self.cfgs_stage3_fuse = cfgs_stage3_fuse
        self.cfgs_stage4 = cfgs_stage4
        self.cfgs_stage5 = cfgs_stage5
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        self.conv = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(input_channel)
        self.act = h_swish()
        # building inverted residual blocks
        block_base = InvertedResidual_ample
        block_refine = InvertedResidual_focal
        
        layers = []
        for k, t, c, use_se, use_hs, s in self.cfgs_head:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_head = nn.Sequential(*layers)
        
        
        ###### stage 1
        layers_stage1 = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage1:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage1.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_stage1 = nn.Sequential(*layers_stage1)
        
        
        ###### stage 2
        input_channel_before = input_channel
        layers_stage2_ample = []
        frame_gen_list_stage2 = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage2_ample:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage2_ample.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            frame_gen_list_stage2.append(navigation(inplanes=input_channel,num_segments=num_segments))
            input_channel = output_channel
        self.list_gen2 = nn.ModuleList(frame_gen_list_stage2)
        self.features_stage2_base = nn.Sequential(*layers_stage2_ample)
        
        layers_stage2_focal = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage2_focal:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel_before * t, 8)
            layers_stage2_focal.append(block_refine(num_segments, input_channel_before, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel_before = output_channel
        input_channel = input_channel_before    
        self.features_stage2_refine = nn.Sequential(*layers_stage2_focal)
        
        layers_stage2_fuse = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage2_fuse:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage2_fuse.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_stage2_fuse = nn.Sequential(*layers_stage2_fuse)
        self.att_gen2 = dynamic_fusion(channel=input_channel, reduction=16)
        
        
        ###### stage 3
        input_channel_before = input_channel
        layers_stage3_ample = []
        frame_gen_list_stage3 = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage3_ample:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage3_ample.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            frame_gen_list_stage3.append(navigation(inplanes=input_channel,num_segments=num_segments))
            input_channel = output_channel
        self.list_gen3 = nn.ModuleList(frame_gen_list_stage3)
        self.features_stage3_base = nn.Sequential(*layers_stage3_ample)
        
        layers_stage3_focal = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage3_focal:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel_before * t, 8)
            layers_stage3_focal.append(block_refine(num_segments, input_channel_before, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel_before = output_channel
        input_channel = input_channel_before    
        self.features_stage3_refine = nn.Sequential(*layers_stage3_focal)
        
        layers_stage3_fuse = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage3_fuse:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage3_fuse.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_stage3_fuse = nn.Sequential(*layers_stage3_fuse)
        self.att_gen3 = dynamic_fusion(channel=input_channel, reduction=16)
        
        
        ###### stage 4
        layers_stage4 = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage4:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage4.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_stage4 = nn.Sequential(*layers_stage4)
        
        ###### stage 5
        layers_stage5 = []
        for k, t, c, use_se, use_hs, s in self.cfgs_stage5:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers_stage5.append(block_base(num_segments, input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features_stage5 = nn.Sequential(*layers_stage5)
        
        # building last several layers
        # self.conv_last = conv_1x1_bn(input_channel, exp_size)
        self.conv_last = nn.Conv2d(input_channel, exp_size, 1, 1, 0, bias=False)
        self.bn_last = nn.BatchNorm2d(exp_size)
        self.act_last = h_swish()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        self.output_channel_num = output_channel[mode]
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.fc = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.5),
            nn.Linear(output_channel, num_class),
        )

        self._initialize_weights()

    def forward(self, x, temperature=1e-8):
        _lists = []
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.features_head[0](x, self.num_segments)
        
        
        for i in range(len(self.features_stage1)):
            x = self.features_stage1[i](x, self.num_segments)
        
        
        x_base = x
        x_refine = x
        for i in range(len(self.features_stage2_base)):
            list_big = self.list_gen2[i](x_base, temperature=temperature)
            _lists.append(list_big)
            x_base = self.features_stage2_base[i](x_base, self.num_segments)
            x_refine = self.features_stage2_refine[i](x_refine, list_big)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        att = self.att_gen2(x_base+x_refine)
        x = self.features_stage2_fuse[0](att*x_base + (1-att)*x_refine, self.num_segments)
        
        
        x_base = x
        x_refine = x
        for i in range(len(self.features_stage3_base)):
            list_big = self.list_gen3[i](x_base, temperature=temperature)
            _lists.append(list_big)
            x_base = self.features_stage3_base[i](x_base, self.num_segments)
            x_refine = self.features_stage3_refine[i](x_refine, list_big)
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        att = self.att_gen3(x_base+x_refine)
        x = self.features_stage3_fuse[0](att*x_base + (1-att)*x_refine, self.num_segments)
        
        
        for i in range(len(self.features_stage4)):
            x = self.features_stage4[i](x, self.num_segments)
        for i in range(len(self.features_stage5)):
            x = self.features_stage5[i](x, self.num_segments)
            
        
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.act_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        
        x = self.fc(x)
        
        return x, _lists
    
    def forward_calc_flops(self, x, temperature=1e-8):
        flops = 0
        _lists = []
        
        c_in = x.shape[1]
        x = self.conv(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv.kernel_size[0] * self.conv.kernel_size[1] / self.conv.groups
        x = self.bn(x)
        x = self.act(x)
        
        x, _flops = self.features_head[0].forward_calc_flops(x, self.num_segments)
        flops += _flops
        
        
        for i in range(len(self.features_stage1)):
            x, _flops = self.features_stage1[i].forward_calc_flops(x, self.num_segments)
            flops += _flops
        
        
        x_base = x
        x_refine = x
        for i in range(len(self.features_stage2_base)):
            list_big, _flops = self.list_gen2[i].forward_calc_flops(x_base, temperature=temperature)
            _lists.append(list_big)
            flops += _flops
            x_base, _flops = self.features_stage2_base[i].forward_calc_flops(x_base, self.num_segments)
            flops += _flops
            x_refine, _flops = self.features_stage2_refine[i].forward_calc_flops(x_refine, list_big)
            flops += _flops
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        att, _flops = self.att_gen2.forward_calc_flops(x_base+x_refine)
        flops += _flops
        x, _flops = self.features_stage2_fuse[0].forward_calc_flops(att*x_base + (1-att)*x_refine, self.num_segments)
        flops += _flops
        
        
        x_base = x
        x_refine = x
        for i in range(len(self.features_stage3_base)):
            list_big, _flops = self.list_gen3[i].forward_calc_flops(x_base, temperature=temperature)
            _lists.append(list_big)
            flops += _flops
            x_base, _flops = self.features_stage3_base[i].forward_calc_flops(x_base, self.num_segments)
            flops += _flops
            x_refine, _flops = self.features_stage3_refine[i].forward_calc_flops(x_refine, list_big)
            flops += _flops
        _,_,h,w = x_refine.shape
        x_base = F.interpolate(x_base, size = (h,w))
        att, _flops = self.att_gen3.forward_calc_flops(x_base+x_refine)
        flops += _flops
        x, _flops = self.features_stage3_fuse[0].forward_calc_flops(att*x_base + (1-att)*x_refine, self.num_segments)
        flops += _flops
        
        
        for i in range(len(self.features_stage4)):
            x, _flops = self.features_stage4[i].forward_calc_flops(x, self.num_segments)
            flops += _flops
            
        for i in range(len(self.features_stage5)):
            x, _flops = self.features_stage5[i].forward_calc_flops(x, self.num_segments)
            flops += _flops
        
        c_in = x.shape[1]
        x = self.conv_last(x)
        flops += c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv_last.kernel_size[0] * self.conv_last.kernel_size[1] / self.conv_last.groups
        x = self.bn_last(x)
        x = self.act_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        
        c_in = x.shape[1]
        x = self.fc(x)
        c_out = x.shape[1]
        flops += c_in * self.output_channel_num + self.output_channel_num * c_out
        
        return x, _lists, self.num_segments * flops

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def af_mobilenetv3(num_segments, num_class, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs_head = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1]]
    cfgs_stage1 = [
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 2]]
    cfgs_stage2_ample = [
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1]]
    cfgs_stage2_focal = [
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1]]
    cfgs_stage2_fuse = [
        [5,   3,  40, 1, 0, 2]]
    cfgs_stage3_ample = [
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1]]
    cfgs_stage3_focal = [
        [3,   6,  80, 0, 1, 1],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1]]
    cfgs_stage3_fuse = [
        [3, 2.3,  80, 0, 1, 1]]
    cfgs_stage4 = [
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1]]
    cfgs_stage5 = [
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]]
    return AFMobileNetV3(num_segments, num_class, cfgs_head, cfgs_stage1, cfgs_stage2_ample, 
                       cfgs_stage2_focal, cfgs_stage2_fuse, cfgs_stage3_ample, cfgs_stage3_focal, 
                       cfgs_stage3_fuse, cfgs_stage4, cfgs_stage5, mode='large', **kwargs)



def AF_mobilenetv3(pretrained=False, path_backbone = '.../.../checkpoint/ImageNet/AF-MobileNetv3.pth.tar', shift=False, num_segments=8, num_class=174, **kwargs):    
    model = af_mobilenetv3(num_segments, num_class)
    if pretrained:
        checkpoint = torch.load(path_backbone, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k[7:] in new_state_dict):
                new_state_dict.update({k[7:]:v})
        model.load_state_dict(new_state_dict)
    return model