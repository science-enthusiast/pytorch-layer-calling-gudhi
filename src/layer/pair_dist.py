import torch

class PairDist(torch.nn.Module):
    def __init__(self):
        """
        -------------------
        Parameters:
            fun : function
                function to compute pairwise distances.
            eps : float
                for avoiding overflof of 1/sqrt(x)
        """
        super(PairDist,self).__init__()
        self.eps = 1e-6

    def forward(self, x):
        numPts, pcDim = x.size()

        #dist01 = self.pairFunc(x[0,:],x[1,:])

        

        a = x.repeat(numPts,1,1)
        c = a.clone()
        b = torch.transpose(c,0,1)
        sub = a-b
        #out = torch.norm(sub,dim=(2,2))
        quad = torch.mean(sub**2, dim=2)
        out = torch.sqrt(quad+self.eps)

        return out




        
