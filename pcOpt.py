#An optimization based modification of a time series through TDE representation.
import torch
from common import convertTde
import numpy as np
from tdeLayerOne import tdeLayerOne
from elemProdLayer import elemProdLayer

#device = torch.device('cpu')
device = torch.device('cuda') #use in case the computer has an Nvidia GPU and CUDA is installed.

N = 5

oneVec = torch.ones(N, device=device, requires_grad = False)
tgtTS = np.random.randn(N)

print(tgtTS)

tgtPC = torch.tensor(convertTde(tgtTS), dtype=torch.float, requires_grad = False)

reqTS = torch.randn(len(tgtTS), device=device, requires_grad=True)

stepSiz = 1e-2

myTdeLayer = tdeLayerOne() 

myElemProdLayer = elemProdLayer()

for t in range(500):
   scaleTS = myElemProdLayer(reqTS, oneVec)
   #print(scaleTS) 
   varPC = myTdeLayer(scaleTS)

#   varPC.requires_grad = True

#   print(varPC.size())

#   print(tgtPC.size())

   loss = (varPC - tgtPC).pow(2).sum()
#   loss = torch.nn.MSELoss()(varPC, tgtPC) 
   print(t, loss.item())

   loss.backward()

   with torch.no_grad():
      reqTS -= stepSiz*reqTS.grad

      reqTS.grad.zero_() 
