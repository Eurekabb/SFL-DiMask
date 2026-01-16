import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import googlenet


def build_googlenet_blocks(pretrained=True, aux_logits=True):
    base_model = googlenet(pretrained=pretrained, aux_logits=aux_logits)

    base_model.aux1 = nn.Identity()
    base_model.aux2 = nn.Identity()

    block1 = nn.Sequential(
        base_model.conv1,
        base_model.maxpool1,
        base_model.conv2,
        base_model.conv3,
        base_model.maxpool2,
    )

    # block2: Inception3
    block2 = nn.Sequential(
        base_model.inception3a,
        base_model.inception3b,
        base_model.maxpool3,
    )

    block3 = nn.Sequential(
        base_model.inception4a,
        base_model.inception4b,
        base_model.inception4c,
        base_model.inception4d,
        base_model.inception4e,
    )

    block4 = nn.Sequential(
        base_model.maxpool4,
        base_model.inception5a,
        base_model.inception5b,
    )

    blocks = nn.ModuleList([block1, block2, block3, block4])
    return base_model, blocks


class GoogLeNet_Client(nn.Module):
    def __init__(self, cut_layer=3, pretrained=True, aux_logits=True):
        super(GoogLeNet_Client, self).__init__()
        assert 1 <= cut_layer <= 4, "cut_layer must be in [1, 4]"
        base_model, blocks = build_googlenet_blocks(pretrained, aux_logits)
        self.blocks = blocks
        self.cut_layer = cut_layer

    def forward(self, x):
        for i in range(self.cut_layer):
            x = self.blocks[i](x)
        return x


class GoogLeNet_Server(nn.Module):
    def __init__(self, num_classes=100, cut_layer=3, pretrained=True, aux_logits=True):
        super(GoogLeNet_Server, self).__init__()
        assert 1 <= cut_layer <= 4, "cut_layer must be in [1, 4]"

        base_model, blocks = build_googlenet_blocks(pretrained, aux_logits)
        self.blocks = blocks
        self.cut_layer = cut_layer

        self.avgpool = base_model.avgpool
        self.dropout = getattr(base_model, "dropout", nn.Identity())

        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        for i in range(self.cut_layer, len(self.blocks)):
            x = self.blocks[i](x)

        x = self.avgpool(x)               # N, C, 1, 1
        x = torch.flatten(x, 1)           # N, C
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GoogLeNet_Full(nn.Module):
    def __init__(self, num_classes=65, pretrained=True, aux_logits=True):
        super(GoogLeNet_Full, self).__init__()
        base_model, blocks = build_googlenet_blocks(pretrained, aux_logits)
        self.blocks = blocks

        self.avgpool = base_model.avgpool
        self.dropout = getattr(base_model, "dropout", nn.Identity())

        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
