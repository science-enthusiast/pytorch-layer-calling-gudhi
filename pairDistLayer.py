import torch

class pairDistLayer(torch.nn.Module):
    def __init__(self, fun):
        """
        fun: function to compute pairwise distances.
        """
        super(pairDistLayer,self).__init__()
        self.pairFunc = fun

    def forward(self, x):
        numPts, pcDim = x.size()

        vecLen = int(numPts*(numPts - 1)/2)

        #lower triangle part of the distance matrix is computed as a flat vector
        pairDistOp = torch.zeros(vecLen) 

        elemCnt = 0

        for iPt in range(1,numPts):
            for jPt in range(iPt):
                pairDistOp[elemCnt] = self.pairFunc(x[iPt,:],x[jPt,:])

                elemCnt += 1

        return pairDistOp 
