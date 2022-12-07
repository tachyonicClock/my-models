import torch
from models.resnet import SmallResNet18 
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as T
import torch.nn as nn
import torch.utils.data
import typing as t
from types import SimpleNamespace
from tqdm import tqdm
import pytorch_lightning as pl
from torchmetrics import Accuracy


# Set up config
cfg = SimpleNamespace()
cfg.batch_size = 128
cfg.learning_rate = 0.001
cfg.epochs = 200

# Set up data
data = SimpleNamespace()
data.root = '/Scratch/al183/datasets/'

# data.dataset = ImageFolder(f"{data.root}/tiny-imagenet-200/train")
data.trainset = CIFAR10(data.root, train=True, download=True)
data.testset = CIFAR10(data.root, train=False, download=True)

# trainset/testset are subsets of the original dataset, so they don't have
# transforms. We need to copy the dataset and set the transforms on the copy.
data.trainset.transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4915, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616))
])
data.testset.transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4915, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616))
])

print(f"Train set size: {len(data.trainset)}")
print(f"Test set size: {len(data.testset)}")

# Set up data loaders
data.trainloader = torch.utils.data.DataLoader(
    data.trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
data.testloader = torch.utils.data.DataLoader(
    data.testset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

#  Set up model

class LightningSmallResNet(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = SmallResNet18(num_classes=200, pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.accuracy(y_hat, y)
        self.log('valid_acc', self.accuracy, on_epoch=True)
        self.log('val_loss', loss)
        return loss


# model = LightningSmallResNet.load_from_checkpoint("lightning_logs/version_25/checkpoints/epoch=15-step=14400.ckpt")
model = LightningSmallResNet()
trainer = pl.Trainer(gpus=1, max_epochs=cfg.epochs)
# Checkpoint lowest loss
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='trained',
    filename='64x64ResNet18CIFAR10-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)
trainer.callbacks.append(checkpoint_callback)
trainer.fit(model, data.trainloader, data.testloader)

# Save the model
torch.save(model.model.state_dict(), 'trained/64x64ResNet18CIFAR10.pkl')
