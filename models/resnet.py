import torch
import torch.nn as nn
import torchvision.models as models

class SmallResNet18(nn.Module):
    """Patch torchvision.models.resnet18 to work with smaller images, such as 
    TinyImageNet (64x64)
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        super(SmallResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained)

        # Patch early layers that overly downsample the image
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.maxpool = nn.Identity()

        # Patch the final layer to output the correct number of classes
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        return self.resnet(x)

