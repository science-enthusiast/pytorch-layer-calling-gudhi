import torch
import collections
from numpy import isinf
import gudhi as gd
from common import attachEdgePairDist

class ripsLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, homDim, maxEdgeLen):
        """
        x: pairwise distances as a flat vector.
        """

        device = torch.device('cpu')

        pairCnt = len(x)

        xNP = x.detach().numpy().copy() #make a numpy copy of the input tensor

        #Lower triangular distance matrix
        distMat = [[]]

        rowSiz = 1
        rowOffset = 0

        while rowSiz + rowOffset < pairCnt + 1:
            curRow = xNP[rowOffset:(rowOffset + rowSiz)].tolist()
            distMat.append(curRow)
            rowOffset += rowSiz
            rowSiz += 1 

 
        ripsComplex = gd.RipsComplex(distance_matrix=distMat, max_edge_length=maxEdgeLen)

        maxDim = homDim + 1

        simplexTree = ripsComplex.create_simplex_tree(max_dimension=maxDim) #considering only one homology dimension

        simplexTree.persistence(homology_coeff_field=2, min_persistence=0)

        persistencePairs = simplexTree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
                                                           #note this array is not alligned with the array of (birth,death) pairs computed by persistence 

        pdSiz = len(persistencePairs)*2 #we are going to create a flat tensor with the birth and death times

        diagTensor = torch.zeros(pdSiz) 

        pdCnt = 0

        for iPair in persistencePairs:

            diagTensor[pdCnt] = simplexTree.filtration(iPair[0]) #append the birth time

            pdCnt += 1

            deathTime = simplexTree.filtration(iPair[1])

            if isinf(deathTime):
                deathTime = maxEdgeLen 

            diagTensor[pdCnt] = deathTime #append the death time

            pdCnt += 1

        derMatTensor = torch.zeros(pairCnt, pdSiz)

        iPD = 0

        for iPair in persistencePairs:

            for iSimplex in iPair:
                if len(iSimplex) > 1:
                    (ind0, ind1) = attachEdgePairDist(x, iSimplex)

                    derMatTensor[int(ind0*(ind0 - 1)/2 + ind1),iPD] = 1
                iPD += 1 

        ctx.derMatTensor = derMatTensor 

        return diagTensor

    @staticmethod 
    def backward(ctx, gradOp):

        return torch.mv(ctx.derMatTensor,gradOp), None, None 
