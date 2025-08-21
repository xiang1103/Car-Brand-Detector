import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self,num_classes:int, 
                 hidden_dim:list[int]):
        super().__init__()
        # start: 3 x 224 x 224 -> 64 x 112x112
        self.conv1=nn.Sequential( nn.Conv2d(
            in_channels=3,
            out_channels= hidden_dim[0],
            kernel_size=3,
            stride=1,
            padding=1 
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2, stride=2 
        )
        )
        
        # 3x112x112 -> 128 x 56x56  
        self.conv2= nn.Sequential(nn.Conv2d(
            in_channels=hidden_dim[0],
            out_channels=hidden_dim[1],
            kernel_size= 3,
            padding=1 
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2, stride=2 
        )
        ) 
        # 32x56x56 -> 256x28x28 
        self.conv3= nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[2],
                kernel_size=3, 
                padding=1 
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2 
            )
        )
        # 256x28x28 -> 128 
        self.fc1= nn.Linear(256*28*28,128)

        # 128 -> num_classes so we can calculate negative log likelihood
        self.fc2= nn.Linear(128,num_classes)
    
    def forward(self,x):
        x= self.conv1(x) 
        x= self.conv2(x) 
        x= self.conv3(x)
        # flatten image
        x= x.view(x.size(0),-1)
        
        x=F.relu(self.fc1(x))
        x=self.fc2(x) 
        return x 
        



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
    

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super().__init__()
        self.inchannel = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 3, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 32, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 64, 2, stride=2) 
        
        self.fc_1= nn.Linear(3136, 128)       
        self.fc = nn.Linear(128, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out=self.fc_1(out)
        out = self.fc(out)
        return out 
         