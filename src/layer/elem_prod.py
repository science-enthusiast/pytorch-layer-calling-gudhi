import torch

class ElemProd(torch.nn.Module):
    def __init__(self):
        super(ElemProd, self).__init__()

    def forward(self, scaleVec, timeSeries):

        return torch.mul(scaleVec, timeSeries)
