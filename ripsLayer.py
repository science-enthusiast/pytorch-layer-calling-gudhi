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

        '''
        print('ripsLayer:forward: x')
        print(x)
        '''

        xNP = x.detach().numpy().copy() #make a numpy copy of the input tensor

        #Lower triangular distance matrix
        distMat = [[]]

        rowSiz = 1
        rowOffset = 0

        while rowSiz + rowOffset < pairCnt + 1:
            curRow = xNP[rowOffset:(rowOffset + rowSiz)].tolist()
            '''
            print('curRow')
            print(curRow)
            '''
            distMat.append(curRow)
            rowOffset += rowSiz
            rowSiz += 1 

        #print('distMat')
        #print(distMat)        
 
        ripsComplex = gd.RipsComplex(distance_matrix=distMat, max_edge_length=maxEdgeLen)
 
        maxDimSimplex = max(homDim) + 1 #considering only sufficient homology dimension 

        simplexTree = ripsComplex.create_simplex_tree(max_dimension=maxDimSimplex)

        simplexTree.persistence(homology_coeff_field=2, min_persistence=0)

        persistencePairs = simplexTree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
                                                           #note this array is not alligned with the array of (birth,death) pairs computed by persistence 
        '''
        print('ripsLayer:forward:persistencePairs')
        print(persistencePairs)
        '''

        pdSiz = dict()

        for iPair in persistencePairs:
            iHomDim = len(iPair[0]) - 1

            if iHomDim in pdSiz.keys():
                pdSiz[iHomDim] += 2
            else:
                pdSiz[iHomDim] = 2

        diagTensorList = []

        pdSizSort = collections.OrderedDict(sorted(pdSiz.items()))

        pdInd = {} #will be useful later, while writing PD values into tensors

        for iHomDim in pdSizSort:
            diagTensorList.append(torch.zeros(pdSizSort[iHomDim], requires_grad=True)) 

            pdInd[iHomDim] = 0 #initialize the indices to zero

        for iPair in persistencePairs:
            iHomDim = len(iPair[0]) - 1

            diagTensorList[list(pdSizSort).index(iHomDim)][pdInd[iHomDim]] = simplexTree.filtration(iPair[0]) #append the birth time

            pdInd[iHomDim] += 1

            deathTime = simplexTree.filtration(iPair[1])

            if isinf(deathTime):
                deathTime = maxEdgeLen 

            diagTensorList[list(pdSizSort).index(iHomDim)][pdInd[iHomDim]] = deathTime #append the death time

            pdInd[iHomDim] += 1

        '''
        print('ripsLayer:forward:diagTensorList')
        print(diagTensorList)
        '''

        derMatListTensor = []

        for iHomDim in range(len(homDim)):
            derMatListTensor.append(torch.zeros(pairCnt, len(diagTensorList[iHomDim]))) 

        iPD = [0 for x in range(len(homDim))]

        for iPair in persistencePairs:
            for iSimplex in iPair:
                if len(iSimplex) > 1:
                    (ind0, ind1) = attachEdgePairDist(x, iSimplex)

                    derMatListTensor[len(iPair[0]) - 1][int(ind0*(ind0 - 1)/2 + ind1),iPD[len(iPair[0]) - 1]] = 1
                iPD[len(iPair[0]) - 1] += 1 

        #ctx.save_for_later(derMatListTensor)
        ctx.derMatListTensor = derMatListTensor 
        ctx.homDim = homDim
        ctx.maxEdgeLen = maxEdgeLen

        diagTensorTuple = () 

        for iDiagTensor in diagTensorList:
            diagTensorTuple += (iDiagTensor,) 

        '''
        print('ripsLayer:forward:diagTensorTuple')
        print(diagTensorTuple)
        '''

        return diagTensorTuple

    @staticmethod 
    def backward(ctx, *gradOp):

        '''
        print('ripsLayer:backward:gradOp')
        print(gradOp)

        print('ripsLayer:backward:derMatListTensor')
        print(ctx.derMatListTensor)
        '''

        iList = 0

        vecSum = torch.zeros(len(ctx.derMatListTensor[0][:,0]),1)

        '''
        print('ripsLayer:backward:vecSum.shape')
        print(vecSum.shape)
        '''

        for iGradOp in gradOp:
            '''
            print('ripsLayer:backward:iGradOp.shape')
            print(iGradOp.shape)
            print('ripsLayer:backward:ctx.derMatListTensor[iList].shape')
            print(ctx.derMatListTensor[iList].shape)
            '''
            vecSum += torch.mm(ctx.derMatListTensor[iList],iGradOp.reshape(len(iGradOp),1))
            iList += 1 

        #print(vecSum)

        return vecSum.reshape(len(ctx.derMatListTensor[0][:,0])), None, None 
