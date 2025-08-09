import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self,num_classes:int, 
                 hidden_dim:list[int]):
        super().__init__()
        # start: 3 x 224 x 224 -> 64 x 
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
        
        # 3x112x112 -> 32 x 56x56  
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
        # 32x56x56 -> 64x28x28 
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
        # 64x28x28 -> 128 
        self.fc1= nn.Linear(64*28*28,128)

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
        
         