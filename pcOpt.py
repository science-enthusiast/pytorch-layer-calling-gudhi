#An optimization based modification of a time series through TDE representation.
import torch
import torch.nn as nn
import torch.optim as optim
from src.layer.util.common import convertTde
import numpy as np
from src.layer.tde import Tde
from src.layer.elem_prod import ElemProd

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

print("device: " + str(device))


class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.TdeLayer = Tde() 

   def forward(self, x):
      return self.TdeLayer(x)

N = 5

tgtTS = np.random.randn(N)
print("tgtTS:" + str(tgtTS))
tgtPC = torch.tensor(convertTde(tgtTS), dtype=torch.float, requires_grad = False)

inputTS = torch.randn(len(tgtTS), device=device, requires_grad=True)


stepSiz = 1e-2

net = Net()
optimizer = optim.Adam([inputTS], lr=stepSiz)

for t in range(500):
   optimizer.zero_grad()
   varPC = net(inputTS)
   loss = (varPC - tgtPC).pow(2).sum()
   # loss = torch.nn.MSELoss()(varPC, tgtPC) 
   print(str(inputTS))
   print(t, loss.item())
   loss.backward()
   optimizer.step()