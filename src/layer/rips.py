import torch
import torch.nn as nn
import collections
from numpy import isinf
import gudhi as gd
from .util.common import attachEdgePairDist

class RipsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hom_dim, max_edge_len):
        """
        x: pairwise distances as a matrix
        ------------------
        Parameters:
            hom_dim : list of intergers
                specify homological dimension to compute
            max_edge_len : float
                maximal edge length for calculation
        """

        device = torch.device('cpu')
        

        #print('ripsLayer:forward: x')
        #print(x)

        x_np = x.detach().numpy().copy() #make a numpy copy of the input tensor
        pairCnt = x.shape[0]

        # #Lower triangular distance matrix
        # distMat = [[]]

        # rowSiz = 1
        # rowOffset = 0

        # while rowSiz + rowOffset < pairCnt + 1:
        #     curRow = xNP[rowOffset:(rowOffset + rowSiz)].tolist()
        #     distMat.append(curRow)
        #     rowOffset += rowSiz
        #     rowSiz += 1 

 
        rips_complex = gd.RipsComplex(distance_matrix=x_np, max_edge_length=max_edge_len)

        max_dim = hom_dim + 1

        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim) #considering only one homology dimension

        simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

        persistence_pairs = simplex_tree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
                                                           #note this array is not alligned with the array of (birth,death) pairs computed by persistence 

        #print('ripsLayer:forward:persistence_pairs')
        #print(persistence_pairs)

        pdSiz = len(persistence_pairs)*2 #we are going to create a flat tensor with the birth and death times

        diagTensor = torch.zeros(pdSiz) 

        pdCnt = 0

        for iPair in persistence_pairs:

            diagTensor[pdCnt] = simplex_tree.filtration(iPair[0]) #append the birth time

            pdCnt += 1

            deathTime = simplex_tree.filtration(iPair[1])

            if isinf(deathTime):
                deathTime = max_edge_len 

            diagTensor[pdCnt] = deathTime #append the death time

            pdCnt += 1

        derMatTensor = torch.zeros(pairCnt, pairCnt, pdSiz)

        iPD = 0

        #import pdb; pdb.set_trace()      


        for iPair in persistence_pairs:
            #print('ripsLayer:forward:iPair')
            #print(iPair)

            for iSimplex in iPair:
                if len(iSimplex) > 1:
                    (ind0, ind1) = attachEdgePairDist(x, iSimplex)

                    #print('ripsLayer:forward: ind0 %d ind1 %d, der. mat. ind: %d, %d' % (ind0, ind1, int(ind0*(ind0 - 1)/2 + ind1), iPD))

                    derMatTensor[ind0,ind1,iPD] = 1
                iPD += 1 

        #print('ripsLayer:forward:derMatTensor')
        #print(derMatTensor)

        ctx.derMatTensor = derMatTensor 

        #import pdb;pdb.set_trace()
        return diagTensor

    @staticmethod 
    def backward(ctx, gradOp):

        #import pdb;pdb.set_trace()
        #print('ripsLayer:backward:gradOp')
        #print(gradOp)

        #print('ripsLayer:backward:derMatTensor')
        #print(ctx.derMatTensor)

        #print('ripsLayer:backward:product')
        #print(torch.mv(ctx.derMatTensor,gradOp))

        return torch.matmul(ctx.derMatTensor,gradOp), None, None 

class Rips(nn.Module):
    def __init__(self, hom_dim, max_edge_len):
        super(Rips, self).__init__()
        self.hom_dim = nn.Parameter(torch.Tensor([hom_dim]), requires_grad=False)
        self.max_edge_len = nn.Parameter(torch.Tensor([max_edge_len]), requires_grad=False)

    def forward(self, x):
        return RipsFunction.apply(x,self.hom_dim, self.max_edge_len)


