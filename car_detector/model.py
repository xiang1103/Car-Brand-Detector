import torch 
import torch.nn as nn 


class CNN(nn.Module):
    def __init__(self,img_h:int, num_layers:int, 
                 hidden_dim:list[int]):
        super().__init__()
        self.conv1= nn.Conv2d(
            in_channels=img_h,
            out_channels= hidden_dim[0],
            kernel_size=2,
            stride=2,
            padding=0 
        )
        self.conv2= nn.Conv2d(
            in_channels=hidden_dim[0],
            out_channels=hidden_dim[1],

        )
    
    def forward(self,x):
        x= self.conv1(x) 
        x= nn.ReLU(x) 
        x= self.conv2(x) 
        x=nn.ReLU(x)  