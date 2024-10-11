import torch.nn as nn
from torchvision.models.video.resnet import (BasicBlock, BasicStem, Conv3DSimple)
from torchvision.models.video.resnet import VideoResNet as TV_VideoResNet, Bottleneck

from mmpretrain.registry import MODELS


@MODELS.register_module()
class VideoResNet(TV_VideoResNet):
    def __init__(self):
        nn.Module.__init__(self)

        block = BasicBlock
        conv_makers = [Conv3DSimple] * 4
        layers = [2, 2, 2, 2]
        stem = BasicStem

        self.inplanes = 64
        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return (x,)