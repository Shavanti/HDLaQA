import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
from DID_module import DualheadCrissCrossAttention

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def ebp(feature1, feature2, a):

    N, C1, H1, W1 = feature1.size()
    N, C2, H2, W2 = feature2.size()

    if (H1 != H2) | (W1 != W2):
        feature2 = F.interpolate(feature2, (H1, W1))

    feature1 = feature1.contiguous().view(N, C1, H1 * W1)
    feature2 = feature2.contiguous().view(N, C2, H1 * W1)
    feature = torch.bmm(feature1, torch.transpose(feature2, 1, 2)) / (H1*W1)  # Bilinear
    assert feature.size() == (N, C1, C2)
    feature = feature.view(N, C1 * C2)
    feature = torch.sqrt(feature + 1e-8)
    feature = pow(a, feature)
    return feature


class HDA(nn.Module):

    def __init__(self):
        super(IDF, self).__init__()

        self.res = resnet50_backbone(pretrained=True).cuda()
        self.l2pool = L2pooling(channels=128)
        self.l2pool_ = L2pooling(channels=16)

        self.DID_1 = DualheadCrissCrossAttention(in_dim=256, out_dim=16)
        self.DID_2 = DualheadCrissCrossAttention(in_dim=144, out_dim=16)
        self.DID_3 = DualheadCrissCrossAttention(in_dim=144, out_dim=16)

        self.fc_1 = nn.Linear(16*16, 64)
        self.fc_1_1 = nn.Linear(64, 1)
        self.fc_2 = nn.Linear(16*16, 64)
        self.fc_2_1 = nn.Linear(64, 1)
        self.fc_3 = nn.Linear(16*16, 64)
        self.fc_3_1 = nn.Linear(64, 1)

        self.a = 2

        self.coef_1 = torch.nn.Parameter(torch.Tensor([0.3]))
        self.coef_2 = torch.nn.Parameter(torch.Tensor([0.3]))
        self.coef_3 = torch.nn.Parameter(torch.Tensor([0.4]))

        self.prelu = nn.PReLU()

        # initialize
        self.DID_1.apply(weights_init_xavier)
        self.DID_2.apply(weights_init_xavier)
        self.DID_3.apply(weights_init_xavier)

        weights_init_xavier(self.fc_1)
        weights_init_xavier(self.fc_2)
        weights_init_xavier(self.fc_3)

    def forward(self, img):

        res_out = self.res(img)

        features_1 = self.l2pool(res_out['features1'])
        features_1_new, features_2 = self.DID_1(features_1, res_out['features2'])  # B, 256, h,w
        fusion_1 = ebp(features_1_new, features_2, self.a)
        F_1 = self.prelu(self.fc_1(fusion_1))
        q_1 = self.fc_1_1(F_1) * self.coef_1

        features_2_new, features_3 = self.DID_2(self.l2pool_(features_2), res_out['features3'])  # B, 256, h,w
        fusion_2 = ebp(features_2_new, features_3, self.a)
        F_2 = self.prelu(self.fc_2(fusion_2))
        q_2 = self.fc_2_1(F_2) * self.coef_2

        features_3_new, features_4 = self.DID_3(self.l2pool_(features_3), res_out['features4'])  # B, 256, h,w
        fusion_3 = ebp(features_3_new, features_4, self.a)
        F_3 = self.prelu(self.fc_3(fusion_3))
        q_3 = self.fc_3_1(F_3) * self.coef_3

        out = q_1 + q_2 + q_3

        return out


class L2pooling(nn.Module):
    """
    l2 pooling module.
    """
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2)//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None]*a[None, :])
        g = g/torch.sum(g)

        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        x = x**2
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out+1e-12).sqrt()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # channel compression module
        self.cc1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))

        self.cc2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))

        self.cc3 = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))

        self.cc4 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))

        weights_init_xavier(self.cc1)
        weights_init_xavier(self.cc2)
        weights_init_xavier(self.cc3)
        weights_init_xavier(self.cc4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        cc_1 = self.cc1(x)

        x = self.layer2(x)
        cc_2 = self.cc2(x)

        x = self.layer3(x)
        cc_3 = self.cc3(x)

        x = self.layer4(x)
        cc_4 = self.cc4(x)

        out = {}
        out['original_feat'] = x
        out['features1'] = cc_1
        out['features2'] = cc_2
        out['features3'] = cc_3
        out['features4'] = cc_4

        return out


def resnet50_backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
