"""
自定义模型
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        pass
    
    def forward(self,x):
        pass 
        return x    