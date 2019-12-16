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
from ripsLayerOneDim import ripsLayer 
from various_func_grad import comp2WassOneDim

#device = torch.device('cpu')
device = torch.device('cuda') #use in case the computer has an Nvidia GPU and CUDA is installed.

N = 100
pcDim = 3
homDim = 0
homDimList = [0]
strictDim = [0]
maxEdgeLen = 10.
 
oneVec = torch.ones(N, device=device, requires_grad = False)
tgtTS = np.random.randn(N)

print(tgtTS)

tgtPC = torch.tensor(convertTde(tgtTS), dtype=torch.float, requires_grad = False)

tgtPDGudhi = tgtRipsPdFromTimeSeries(tgtTS, pcDim, homDimList, maxEdgeLen) 

tgtPDList = gudhiToTensorList(tgtPDGudhi, homDimList, maxEdgeLen)

tgtPD = tgtPDList[0]

reqTS = torch.randn(len(tgtTS), device=device, requires_grad=True)

stepSiz = 1e-2

myRipsLayerApply = ripsLayer.apply

myPairDistLayer = pairDistLayer(compPairDist)

myTdeLayer = tdeLayerOne() 

myElemProdLayer = elemProdLayer()

for t in range(1000):
   scaleTS = myElemProdLayer(reqTS, oneVec)
   #print(scaleTS) 
   varPC = myTdeLayer(scaleTS)
   pairDistVec = myPairDistLayer(varPC) 
   varPD = myRipsLayerApply(pairDistVec, homDim, maxEdgeLen)

   '''
   print('tgtPD')
   print(tgtPD)

   print('varPD')
   print(varPD)
   '''

   #loss = torch.sum(pairDistVec)

   loss = comp2WassOneDim(varPD, tgtPD) 

   #loss = (varPC - tgtPC).pow(2).sum()

   #loss = varPC[0,0]

   #loss = varPD[0][0]

   '''
   for p in range(int(len(varPD)/2)):
      loss += varPD[2*p]
      if varPD[2*p + 1] != maxEdgeLen:
          loss += varPD[2*p + 1]

   loss = torch.tensor(loss)
   '''

   print('iteration %d loss %f' % (t,loss.item()))
 
   loss.backward()

   with torch.no_grad():
      reqTS -= stepSiz*reqTS.grad

      reqTS.grad.zero_() 
