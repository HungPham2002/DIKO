import torch
import torch.nn as nn
import torchvision.models as models

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        
    def forward(self,x):
        x = self.inception._transform_input(x)
        x, aux = self.inception._forward(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet201(weights='IMAGENET1K_V1')
        
    def forward(self, x):
        x = self.densenet(x)
        return x
    
class DIKO(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = InceptionV3()
        self.dense = DenseNet201()
        self.fc1 = nn.Linear(1000+1000, 512)
        self.relu1  = nn.ReLU()
        self.fc2 = nn.Linear(512,128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,bs1, bs2):
        bs1 = self.inception(bs1)
        bs2 = self.dense(bs2)
        x = torch.cat((bs1, bs2), 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x