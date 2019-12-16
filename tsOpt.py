#A toy optimization problem. TS means time series. But it is not relevant. It is just a 1-D vector of values.

import torch
import numpy as np
from elemProdLayer import elemProdLayer

device = torch.device('cpu')
#device = torch.device('cuda') #use in case the computer has an Nvidia GPU and CUDA is installed.

N = 5

oneVec = torch.ones(N, requires_grad = False)

reqTS = torch.randn(N, device=device, requires_grad = True)

tgtTS = torch.randn(N, device=device, requires_grad = False)

print(tgtTS)

stepSiz = 1e-2

myElemProdLayer = elemProdLayer()

for t in range(500):
   print(reqTS)
   scaleTS = myElemProdLayer(reqTS, oneVec)

   scaleTS = torch.tensor(scaleTS, requires_grad = True)
   #print(scaleTS) 

   loss = torch.nn.MSELoss()(tgtTS, scaleTS) 
   print(t, loss.item())

   loss.backward()

   with torch.no_grad():
      reqTS -= stepSiz*reqTS.grad

      reqTS.grad.zero_() 
