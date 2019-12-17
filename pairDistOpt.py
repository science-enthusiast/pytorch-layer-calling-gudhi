import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from src.layer.tde import Tde
#from layer.elemprodlayer import ElemProd
from src.layer.pair_dist import PairDist

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

print("device: " + str(device))

N = 10
pcDim = 3
homDim = [0]
strictDim = [0]
maxEdgeLen = 10.

class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.PairDist = PairDist()
      self.TdeLayer = Tde() 

   def forward(self, x):
      x = self.TdeLayer(x)
      x = self.PairDist(x)
      return x
      

inputTS = torch.randn(N, device=device, requires_grad=True)

stepSiz = 1e-3

net = Net()
optimizer = optim.Adam([inputTS], lr=stepSiz)


def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    #import pdb; pdb.set_trace()


net.PairDist.register_backward_hook(printgradnorm)
net.TdeLayer.register_backward_hook(printgradnorm)



for t in range(500):
   optimizer.zero_grad()
   pairDistVec = net(inputTS) 
   loss = torch.sum(pairDistVec)
   #import pdb;pdb.set_trace()
   print("%d %f" % (t,loss)) 
   loss.backward()
   optimizer.step()
   #net.zero_grad()
   