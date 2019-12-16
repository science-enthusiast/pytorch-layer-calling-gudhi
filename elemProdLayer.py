import torch

class elemProdLayer(torch.nn.Module):
    def __init__(self):
        super(elemProdLayer, self).__init__()

    def forward(self, scaleVec, oneVec):

        self.oneVec = oneVec

        return torch.mul(scaleVec, oneVec)

    def backward(self, opVec):
        return torch.mul(self.oneVec, opVec)
