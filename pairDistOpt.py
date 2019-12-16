#An optimization based modification of a time series through TDE representation.
import torch
from common import convertTde
from common import compPairDist
from common import tgtRipsPdFromTimeSeries
from common import gudhiToTensorList
import numpy as np
from tdeLayerOne import tdeLayerOne
from elemProdLayer import elemProdLayer
from pairDistLayer import pairDistLayer
from ripsLayer import ripsLayer 
from various_func_grad import comp2Wass

if torch.cuda.is_available():
   print('CUDA IS AVAILABLE')
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

print("device: " + str(device))

#device = torch.device('cpu')
#device = torch.device('cuda') #use in case the computer has an Nvidia GPU and CUDA is installed.

N = 5
pcDim = 3
homDim = [0]
strictDim = [0]
maxEdgeLen = 10.
 
oneVec = torch.ones(N, requires_grad = False)

reqTS = torch.randn(N, device=device, requires_grad=True)

stepSiz = 1e-3

myPairDistLayer = pairDistLayer(compPairDist)

myTdeLayer = tdeLayerOne() 

myElemProdLayer = elemProdLayer()

for t in range(500):
   scaleTS = myElemProdLayer(reqTS, oneVec)
   print(scaleTS) 
   varPC = myTdeLayer(scaleTS)
   pairDistVec = myPairDistLayer(varPC) 

   loss = torch.sum(pairDistVec)   
   print('loss')
   print(t, loss.item())
 
   loss.backward()

   with torch.no_grad():
      reqTS -= stepSiz*reqTS.grad

      reqTS.grad.zero_() 
