"""Define the model architecture."""




# pylint: disable=unsubscriptable-object,too-many-instance-attributes
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
import torch.nn as nn

class MobileNetV2Config(nn.Module):
    def __init__(self):
        super(MobileNetV2Config, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            nn.Conv2d(32, 16, 1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            nn.InvertedResidual(16, 96, 1, 6),
            nn.InvertedResidual(96, 144, 2, 6),
            nn.InvertedResidual(144, 24, 2, 6),
            nn.InvertedResidual(24, 144, 2, 6),
            nn.InvertedResidual(144, 32, 2, 6),
            nn.InvertedResidual(32, 192, 1, 6),
            nn.InvertedResidual(192, 192, 2, 6),
            nn.InvertedResidual(192, 48, 2, 6),
            nn.InvertedResidual(48, 192, 2, 6),
            nn.InvertedResidual(192, 64, 2, 6),
            nn.InvertedResidual(64, 384, 1, 6),
            nn.InvertedResidual(384, 384, 1, 6),
            nn.InvertedResidual(384, 64, 1, 6),
            nn.InvertedResidual(64, 384, 1, 6),
            nn.InvertedResidual(384, 96, 1, 6),
            nn.InvertedResidual(96, 576, 1, 6),
            nn.InvertedResidual(576, 160, 2, 6),
            nn.InvertedResidual(160, 960, 1, 6),
            nn.Conv2d(960, 1280, 1, stride=1, padding=0),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x


def create_cnn_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for param in model.parameters():
        param.requires_grad = True
    
    model.classifier = nn.Linear(1280, 2)
    CNNModel = model.to(device)
    
    return CNNModel