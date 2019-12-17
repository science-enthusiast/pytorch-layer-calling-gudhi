#An optimization based modification of a time series through TDE representation.
import torch
import torch.nn as nn
import torch.optim as optim
from src.layer.util.common import convertTde
# from common import compPairDist
from src.layer.util.common import tgtRipsPdFromTimeSeries
from src.layer.util.common import gudhiToTensorList
import numpy as np
from src.layer.tde import Tde
from src.layer.elem_prod import ElemProd
from src.layer.pair_dist import PairDist
from src.layer.rips import Rips
from src.layer.util.various_func_grad import comp2WassSingleDim

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

print("device: " + str(device))


N = 30
pcDim = 3
homDim = 0
homDimList = [0,1]
strictDim = 0
strictDimList = [0]
maxEdgeLen = 10.


class Net(nn.Module):
   def __init__(self, hom_dim, max_edge_len):
      super(Net, self).__init__()
      self.TdeLayer = Tde() 
      self.PairDist = PairDist()
      self.Rips = Rips(hom_dim, max_edge_len)

   def forward(self, x):
      x = self.TdeLayer(x)
      x = self.PairDist(x)
      x = self.Rips(x)
      return x

tgtTS = np.random.randn(N)
print(tgtTS)

tgtPC = torch.tensor(convertTde(tgtTS), dtype=torch.float, requires_grad = False)
tgtPDGudhi = tgtRipsPdFromTimeSeries(tgtTS, pcDim, homDimList, maxEdgeLen) 
tgtPDList = gudhiToTensorList(tgtPDGudhi, homDimList, maxEdgeLen)
tgtPD = tgtPDList[0]

inputTS = torch.randn(len(tgtTS), device=device, requires_grad=True)

stepSiz = 1e-1
net = Net(homDim, maxEdgeLen)
optimizer = optim.Adam([inputTS], lr=stepSiz)



for t in range(500):
   optimizer.zero_grad()
   varPD = net(inputTS)
   
   print('tgtPD')
   print(tgtPD)

   print('varPD')
   print(varPD)

   loss = comp2WassSingleDim(varPD, tgtPD) 
   print('loss')
   print(t, loss.item())
 
   loss.backward()
   optimizer.step()
   