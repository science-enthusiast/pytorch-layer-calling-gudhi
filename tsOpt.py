#An optimization based modification of a time series through TDE representation.
import torch
from src.layer.util.common import convertTde
import numpy as np
from src.layer.tde import Tde
from src.layer.elem_prod import ElemProd

device = torch.device('cpu')
#device = torch.device('cuda') #use in case the computer has an Nvidia GPU and CUDA is installed.

N = 5

oneVec = torch.ones(N, requires_grad = False)

reqTS = torch.randn(N, device=device, requires_grad = True)

tgtTS = torch.randn(N, device=device, requires_grad = False)

print(tgtTS)

stepSiz = 1e-2

ElemProd = ElemProd()

for t in range(500):
   print(reqTS)
   scaleTS = ElemProd(reqTS, oneVec)
   #print(scaleTS) 

   loss = torch.nn.MSELoss()(tgtTS, scaleTS) 
   print(t, loss.item())

   loss.backward()

   with torch.no_grad():
      reqTS -= stepSiz*reqTS.grad

      reqTS.grad.zero_() 
