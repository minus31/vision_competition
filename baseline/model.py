import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        # model = models.resnet18(pretrained=True)
        model = models.densenet161(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class DenseNet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.densenet161(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model = list(model.children())[:-1]
        model.append(nn.Conv2d(2208, out_size, 7))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

from efficientnet_pytorch import EfficientNet


# class Efficientnet(nn.Module):
#     def __init__(self, out_size):
#         super().__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b7')
        
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#     def forward(self, image):
    
#         f = self.model.extract_features(image)
#         conv = nn.Conv2d(2560, 350, 7).cuda()
#         f = conv(f)

#         return f.squeeze(-1).squeeze(-1)

#############################################
# no update extractor and add FCN at the last 

class Efficientnet(nn.Module):

    def __init__(self, out_size):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.model._fc = nn.Linear(2560, 350)
        
    def forward(self, image):
        return self.model(image)