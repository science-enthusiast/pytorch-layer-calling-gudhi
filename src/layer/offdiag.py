from torch import nn 
import torch

class OffDiag(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        return  x - torch.diag(torch.diag(x))
