"""
Created on 30/1/2019

@author Hariprasad Kannan - Inria DataShape

All rights reserved
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import optimize as spopt
from lapsolver import solve_dense
from sklearn.neighbors import KDTree
import gudhi as gd

"""
A pair of points in Euclidean space is given, for which,
the pairwise Euclidean distance is computed.
"""

def compPairDist(ptOne, ptTwo):
    """
    Input:
    ptOne and ptTwo: Two input points in the form of 2D NumPy array, where
                     number of columns is equal to the dimension of the space.
    
    Output:
    eucDist: Pairwise Euclidean distances in the form of list of lists.
    """

    curDist = ptOne - ptTwo
    curDist = curDist*curDist #using numpy.multiply() leads to RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    curDist = torch.sum(curDist)
    return torch.sqrt(curDist) #as above, numpy.sqrt() cannot be used.

"""
A set of points in Euclidean space is given, for which,
pairwise Euclidean distances are computed.
"""

def EuclideanDistances(points):
    """
    Input:
    points: Input points in the form of a 2D NumPy array, where
            number of rows is equal to number of points.
    
    Output:
    eucDist: Pairwise Euclidean distances in the form of list of lists.
    """
    nPts = points.shape[0]

    eucDist = []
    for i in range(nPts):
        eucDist_i = []
        for j in range(i):
            curDist = points[i,:] - points[j,:]
            curDist = np.multiply(curDist, curDist)
            curDist = sum(curDist)
            curDist = np.sqrt(curDist)
            eucDist_i.append(curDist)
        eucDist.append(eucDist_i)

    return eucDist

#==============================================================================
# Functions to compute the DTM
#==============================================================================

# Obtain the k nearest neighbours for a set of points, from another set of points.

def get_kNN(X,queryPts, k):
    '''
    Input:
    X: a nxd numpy array representing n points in R^d
    queryPts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)

    Output: (dist,ind)
    dist: a mxk numpy array where each row contains the (Euclidean) distance to
          the first k NNs to the corresponding row of queryPts.
    ind: a mxk numpy array where each row contains the indices of the k NNs in
         X of the corresponding row in queryPts
    '''
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    dist, ind = kdt.query(queryPts, k, return_distance=True)

    return (dist, ind)

"""
Determine distance to measure (DTM) based weights for the points in a point cloud.
Intuitively, points in highly dense region get less weight and outliers get more weight. 
"""

def DTM(X,queryPts,k):
    '''
    Input:
    X: a nxd numpy array representing n points in R^d
    queryPts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)

    Output:
    DTM_result: a mx1 numpy array containg the DTM (with exponent p=2) to the
    query points.

    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTMs = DTM(X,Q,3)
    '''
    nnDist, NN = get_kNN(X,queryPts,k)
    dtmResult = np.sqrt(np.sum(nnDist*nnDist,axis=1) / k)

    return(dtmResult)

def DTM_revised(X,queryPts,k):
    '''
    Input:
    X: a nxd numpy array representing n points in R^d
    queryPts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)

    Output:
    DTM_result: a mx1 numpy array containg the DTM (with exponent p=2) to the
    query points.

    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTMs = DTM(X,Q,3)
    '''
    nnDist, NN = get_kNN(X,queryPts,k)
    dtmResult = np.sqrt(np.sum(nnDist*nnDist,axis=1) / k)

    return(dtmResult,nnDist,NN)

#==============================================================================
#Functions to compute filtrations
#==============================================================================
def StructureW(X, F, distances, edge_max, dimension_max = 2):
    '''
    Compute the Rips-W filtration of a point cloud, weighted with the DTM
    values
        st = StructureW(X, F, distances, dimension_max = 2)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + F: the values of a function over the set X
    + dim: the dimension of the skeleton of the Rips (dim_max = 1 or 2)
    Output:
    + st: a gd.SimplexTree with the constructed filtration (require Gudhi)
    '''
    nPts = X.shape[0]

    alpha_complex = gd.AlphaComplex(points=X)
    st = alpha_complex.create_simplex_tree()
    stDTM = gd.SimplexTree()
    for simplex in st.get_filtration():
        if len(simplex[0])==1:
            i = simplex[0][0]
            stDTM.insert([i], filtration  = F[i])
        if len(simplex[0])==2:
            i = simplex[0][0]
            j = simplex[0][1]
            if (j < i):
                filtr = (distances[i][j] + F[i] + F[j])/2
            else:
                filtr = (distances[j][i] + F[i] + F[j])/2

            stDTM.insert([i,j], filtration  = filtr)
    stDTM.expansion(dimension_max)
    st = stDTM

    """
    st = gd.SimplexTree()
    for i in range(nPts):
        st.insert([i], filtration = F[i])

    for i in range(nPts):
        for j in range(i):
            if distances[i][j]<edge_max:
                val = (distances[i][j] + F[i] + F[j])/2
                filtr = max([F[i], F[j], val])
                st.insert([i,j], filtration  = filtr)

    st.expansion(dimension_max)
    """

    result_str = 'Complex W is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.'

    return st

#==============================================================================
#Functions to make various plots.
#==============================================================================

#Plot the elementwise difference between two time series.

def makeDiffPlot(firstTS, secondTS):
   """
   Input: Two time series.
   Output: Point-wise difference between the two time series.
           The list is plotted and returned.
   """ 
   if len(firstTS) != len(secondTS):
      print("The two time series do not have same length\n")
      return

   diffList = [x-y for (x,y) in zip(firstTS, secondTS)]

   plt.plot(diffList)
   plt.show()

   return diffList

#Plot persistence diagram by treating point at infinity in a suitable manner.

def plotPdOwnVersion(pdIp, alpha=0.6, infty=2.0,label=""):
    (min_birth, max_death) = gd.__min_birth_max_death(pdIp)
    ind = 0

    print("max_death %f min_birth %f" % (max_death, min_birth))

    delta = ((max_death - min_birth) / 100.0)
    infinity = infty

    x = np.linspace(0, infinity + delta, 1000)

    plt.plot(x, [infinity] * len(x), linewidth=1.0, color='k', alpha=alpha)
    plt.plot(x, x, linewidth=1.0, color='k', alpha=alpha)
    plt.text(0, infinity, r'$\infty$', color='k', alpha=alpha)

    maxDeath = 0
    minBirth = 1

    for interval in pdIp:
        print("x %f y %f" % (interval[1][0], interval[1][1]))

        if interval[1][0] < minBirth:
            minBirth = interval[1][0]

        if float(interval[1][1]) != float('inf'):
            if interval[1][1] > maxDeath:
                maxDeath = interval[1][1] 

            #Plot finite death case
            plt.scatter(interval[1][0], interval[1][1], alpha=alpha, color=gd.palette[int(interval[0])])
        else:
            #Plot infinite death case at a reasonable location, so that the diagram looks nicer.
            plt.scatter(interval[1][0], infinity, alpha=alpha, color=gd.palette[int(interval[0])])

    print("plotted values: maxDeath %f minBirth %f" % (maxDeath, minBirth))

    print("infinity %f delta %f infinity + delta %f" % (infinity, delta, infinity + delta))

    plt.title('Persistence diagram'+label)
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.axis([0, infinity + delta, 0, infinity + delta])
    plt.show()

#Plot matching points between two PDs for a given homology dimension.
 
def plotMatch(pdOne, pdTwo, homDim, one2One = False, maxEdgeLen = 10.0):
    """
    Input: pdOne, pdTwo: persistence diagrams, pdOne and pdTwo.
           homDim: given homology dimension.
           one2One: flag indicating whether one-to-one match is enforced.
           maxEdgeLen: Maximum edge length of the filtration. 
    Output: Plot indicating matching points between the two PDs.
            Squared 2-Wasserstein distance based cost. 
    """
    if (one2One):
    #for 0-homology, the points are matched in an on-to-one manner.
    #note: the number of points should be equal between the two PDs.
        pdAugTwo = list(pdTwo)
        matchPts = findMatchConstrained(pdOne, pdTwo, homDim)
    else:
    #otherwise, each PD is augmented by the projection on to the diagonal from the other PD.
        pdAugOne, pdAugTwo, numPtsOne, numPtsTwo = augmentPd(pdOne, pdTwo, homDim) 
        matchPts = findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, homDim)

    pdOnePick = [x for x in pdOne if (x[0] == homDim)]
    pdAugTwoPick = [x for x in pdAugTwo if (x[0] == homDim)]

    numPtsOne = len(pdOnePick)

    plotData = []

    funcVal = 0

    for iPtPD in range(numPtsOne):
        #for 0-homology, the point with maximum persistence will have death time equal to maximum edge length.
        #This point exists in each PD and these points get matched between the two PDs. For better visualization this pair of points are not plotted. 
        if (homDim != 0) or ((homDim == 0) and (pdOnePick[iPtPD][1][1] != maxEdgeLen) and (pdAugTwoPick[matchPts[iPtPD]][1][1] != maxEdgeLen)):
            plotData.extend([(pdOnePick[iPtPD][1][0],pdAugTwoPick[matchPts[iPtPD]][1][0]),(pdOnePick[iPtPD][1][1],pdAugTwoPick[matchPts[iPtPD]][1][1]), 'r'])
            funcVal = funcVal + (pdOnePick[iPtPD][1][0] - pdAugTwoPick[matchPts[iPtPD]][1][0])**2 + (pdOnePick[iPtPD][1][1] - pdAugTwoPick[matchPts[iPtPD]][1][1])**2

    numPtsOneTotal = len(pdAugOne)

    #taking care of points in pdTwo that are matched to their projection on the diagonal
    for iPtPd in range(numPtsOne,numPtsOneTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
        if (matchPts[iPtPd] < numPtsTwo):
            diagVal = (pdAugTwoPick[matchPts[iPtPd]][1][0] + pdAugTwoPick[matchPts[iPtPd]][1][1])/2
            plotData.extend([(pdAugTwoPick[matchPts[iPtPD]][1][0], diagVal),(pdAugTwoPick[matchPts[iPtPD]][1][1], diagVal), 'r'])
            funcVal = funcVal + (pdAugTwoPick[matchPts[iPtPd]][1][0] - diagVal)**2 + (pdAugTwoPick[matchPts[iPtPd]][1][1] - diagVal)**2
 
    print("Squared 2-Wasserstein cost: %f" % (funcVal))

    plt.figure()
    plt.plot(*plotData)
    plt.axis('equal')
    plt.show()

"""
Obtain target PD from input PD through some rules.
Currently, rules for homology dimensions 0 and 1 are implemented.
We are referring to the set of rules implemented in this function as RuleOne.
The input PD could be obtained through any filtration: Rips, DTM etc.  
"""

def tgtPdThruRuleOne(ipPD,maxEdgeLen,homDim):
    #homDim is a list indicating which homology dimensions are being considered.

    #target PD is a list of numpy arrays
    tgtPD = []

    iCnt = 0

    maxPers = 0 #keep track of point with maximum persistence 
    maxInd = 0  #in 1-homology

    for iDiag in ipPD:
        #target PD for homology dimension 0, based on the following rule:
        #pull all points with death time greater than a threshold towards that threshold (except point at infinity)

        threshVal = 1 #used by 0-homology to move points according to threshold

        birthTime = iDiag[1][0] #birth time and death time
        deathTime = iDiag[1][1] #to be modified by the following logic

        if (iDiag[0] == 0) and (0 in homDim):

            #for point at infinity, create point with death time equal to max edge length
            if (np.isinf(deathTime)):
                deathTime = maxEdgeLen

            elif deathTime > threshVal:
                deathTime = threshVal 
                if birthTime > threshVal: #if birth time is greater than the threshold, change its value to the threshold. 
                    birthTime = threshVal #hence, we have changed the point to a point on the diagonal. 

            #leave other points undisturbed

            tgtPD.append((0, (birthTime,deathTime)))
        
        #for the rule for 1-homology, first identify point with maximum persistence. 
        elif (iDiag[0] == 1) and (1 in homDim):

            if (deathTime-birthTime > maxPers):
                maxPers = deathTime-birthTime
                maxInd = iCnt
 
            iCnt = iCnt + 1

    #rule for 1-homology

    iCnt = 0

    for iDiag in ipPD:         
        if (iDiag[0] == 1) and (1 in homDim):
            birthTime = iDiag[1][0] #birth time and death time
            deathTime = iDiag[1][1] #to be modified by the following logic

            if (iCnt == maxInd):
                birthTime += 0       #for the point with maximum persistence
                deathTime += 0.5 #modify birth and death times by offsets 

            tgtPD.append((1,(birthTime,deathTime)))
 
            iCnt = iCnt + 1

    return tgtPD

"""
Obtain target persistence diagram (DTM based) from a given time series.
"""
def tgtDtmPdFromTimeSeries(tgtTimeSeries, pcDim, homDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1):
    #homDim is a list indicating which homology dimensions are being considered.

    pcPts = convertTde(tgtTimeSeries, skip = tdeSkip, delay = tdeDelay, dimension = pcDim) #create point cloud (PC) for time series 

    print("tgtDtmPdFromTimeSeries: no. of points "),
    print(pcPts.shape)  

    #Build DTM based filtration
    dtmValues = DTM(pcPts,pcPts,kNN)
    distances = EuclideanDistances(pcPts)
    simplex_tree = StructureW(pcPts,dtmValues,distances,edge_max=maxEdgeLen)

    #PD of the target PC
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

    diagPlt = gd.plot_persistence_diagram(diag)

    #diagPlt.axis('equal')
    diagPlt.show()
 
    tgtPD = [] #target PD is a list of tuples, same as Gudhi format.

    maxPers = 0
    maxInd = 0

    dimZeroCnt = 0

    for iDiag in diag:
        if (iDiag[0] in homDim):
            if (iDiag[0] == 0):
                dimZeroCnt += 1

            birthTime = iDiag[1][0]
            deathTime = iDiag[1][1] 

            if (np.isinf(deathTime)):
                deathTime = maxEdgeLen

            tgtPD.append((iDiag[0],(birthTime,deathTime)))

    print("No. of points in target PD %d, no. of zero dimensional points %d" % (len(tgtPD), dimZeroCnt))

    return tgtPD

"""
Obtain target persistence diagram (Rips based) from a given time series.
"""
def tgtRipsPdFromTimeSeries(tgtTimeSeries, pcDim, homDim, maxEdgeLen):
    #homDim is a list indicating which homology dimensions are being considered.

    pcPts = convertTde(tgtTimeSeries,dimension=pcDim) #create point cloud (PC) for time series 

    maxDimSimplex = max(homDim) + 1 #considering only sufficient homology dimension 

    rips_complex = gd.RipsComplex(pcPts, max_edge_length=maxEdgeLen)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxDimSimplex)

    #PD of the target PC
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

    print('tgtRaipsPdFromTimeSeries:diag')
    print(diag) 
 
    tgtPD = [] #target PD is a list of tuples, same as Gudhi format.

    maxPers = 0
    maxInd = 0

    for iDiag in diag:
        if (iDiag[0] in homDim):
            birthTime = iDiag[1][0]
            deathTime = iDiag[1][1] 

            if (np.isinf(deathTime)):
                deathTime = maxEdgeLen

            tgtPD.append((iDiag[0],(birthTime,deathTime)))

    return tgtPD
 
"""
Augment each PD with projection onto the diagonal of the points of the other PD, for the given homology dimension.
"""
def augmentPd(pdOne, pdTwo, homDim):
#pdOne and pdTwo are persistence diagrams, according to Gudhi format.
#They are a list of tuples, where each tuple is composed of a scalar and a tuple of two numbers.
#The scalar is the homology dimension and the two numbers are birth and death times. 

    pdOnePick = [x for x in pdOne if (x[0] == homDim)]
    pdTwoPick = [x for x in pdTwo if (x[0] == homDim)]

    numPtsOne = len(pdOnePick)
    numPtsTwo = len(pdTwoPick)

    pdAugOne = list(pdOne)
    pdAugTwo = list(pdTwo)

    for iPt in pdOnePick:
        projVal = (iPt[1][0] + iPt[1][1])/2.0
        pdAugTwo.append((homDim,(projVal,projVal))) 

    for iPt in pdTwoPick:
        projVal = (iPt[1][0] + iPt[1][1])/2.0
        pdAugOne.append((homDim,(projVal,projVal)))

    return pdAugOne, pdAugTwo, numPtsOne, numPtsTwo
 
"""
Augment each PD with projection onto the diagonal of the points of the other PD, for the given homology dimension.
Each PD is a tensor, corresponding to points of a particular homology dimension.
"""
def augmentPdTensor(pdOne, pdTwo):
#pdOne and pdTwo are flat tensors. For a particular homology dimension. The birth time follows the death time for each point in the PD. 

    numPtsOne = int(0.5*len(pdOne))
    numPtsTwo = int(0.5*len(pdTwo))

    extElems = torch.zeros(2*numPtsOne)

    indElem = 0

    for iPt in range(numPtsOne):
        projVal = 0.5*(pdOne[2*iPt] + pdOne[2*iPt + 1])
        extElems[indElem] = projVal
        indElem += 1
        extElems[indElem] = projVal
        indElem += 1

    pdTwo = torch.cat((pdTwo,extElems)) 

    extElems = torch.zeros(2*numPtsTwo)

    indElem = 0

    for iPt in range(numPtsTwo):
        projVal = 0.5*(pdTwo[2*iPt] + pdTwo[2*iPt + 1])
        extElems[indElem] = projVal
        indElem += 1
        extElems[indElem] = projVal
        indElem += 1

    pdOne = torch.cat((pdOne,extElems))

    return pdOne, pdTwo, numPtsOne, numPtsTwo

"""
Find 2-Wasserstein distance based matching, between two PDs for a given homology dimension.
The set of points for each PD is augmented with diagonal projection of points from the other PD. 
""" 
def findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, homDim, solver="scipy"):
#pdAugOne and pdAugTwo are persistence diagrams, which are augmented by projections on the diagonal
#of points from the other PD, for the given homology dimension. They are usually created by the function augmentPd.  

    #numPtsOne is the number of points in pdOne for given homology dimension before augmentation.
    #numPtsTwo is the number of points in pdTwo for given homology dimension before augmentation.
    totPts = numPtsOne + numPtsTwo

    #the main aspect is about defining the cost matrix

    costMat = np.zeros((totPts, totPts),dtype=np.float64)

    pdAugOnePick = [x for x in pdAugOne if (x[0] == homDim)]
    pdAugTwoPick = [x for x in pdAugTwo if (x[0] == homDim)] 

    for iPt in range(totPts):
        for jPt in range(totPts):
            #pairwise costs between original (not augmented) points 
            if (iPt < numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOnePick[iPt][1][0] - pdAugTwoPick[jPt][1][0])**2 + (pdAugOnePick[iPt][1][1] - pdAugTwoPick[jPt][1][1])**2

            #for a given original point in a PD, pairwise costs w.r.t. to all the augmented points in the other PD
            #is the same and it is equal to the cost w.r.t. to that points projection on the diagonal
            elif (iPt < numPtsOne) and (jPt >= numPtsTwo):
                costMat[iPt,jPt] = (pdAugOnePick[iPt][1][0] - pdAugTwoPick[numPtsTwo + iPt][1][0])**2 + (pdAugOnePick[iPt][1][1] - pdAugTwoPick[numPtsTwo + iPt][1][1])**2 

            elif (iPt >= numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOnePick[numPtsOne + jPt][1][0] - pdAugTwoPick[jPt][1][0])**2 + (pdAugOnePick[numPtsOne + jPt][1][1] - pdAugTwoPick[jPt][1][1])**2 

            #the last case is (iPt >= numPtsOne) and (jPt >= numPtsTwo), for which cost between projected points on the diagonal is zero

    if (solver == "scipy"): 
        srcInd, tgtInd = spopt.linear_sum_assignment(costMat)
    elif (solver == "solve_dense"):
        srcInd, tgtInd = solve_dense(costMat)  

    #if an original point is mapped to one of the projections on the diagonal,
    #then that point is mapped to its projection on the diagonal. 
    for iPt in range(numPtsOne):
        if (tgtInd[iPt] >= numPtsTwo):
            tgtInd[iPt] = numPtsTwo+iPt  

    for iPt in range(numPtsTwo):
        if (srcInd[iPt] >= numPtsOne):
            tgtInd[iPt] = numPtsTwo+iPt  

    return tgtInd

"""
Find 2-Wasserstein distance based matching, between two PDs of a given homology dimension.
The set of points for each PD is augmented with diagonal projection of points from the other PD.
The PDs are given as flat tensors. 
""" 
def findMatchTensor(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, solver="scipy"):
#pdAugOne and pdAugTwo are persistence diagrams, which are augmented by projections on the diagonal
#with points from the other PD, for the given homology dimension. They are usually created by the function augmentPdTensor.  

    #numPtsOne is the number of points in pdOne for given homology dimension before augmentation.
    #numPtsTwo is the number of points in pdTwo for given homology dimension before augmentation.
    totPts = numPtsOne + numPtsTwo

    #the main aspect is about defining the cost matrix

    costMat = np.zeros((totPts, totPts),dtype=np.float64)

    for iPt in range(totPts):
        for jPt in range(totPts):
            #pairwise costs between original (not augmented) points 
            if (iPt < numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[2*iPt] - pdAugTwo[2*jPt])**2 + (pdAugOne[2*iPt + 1] - pdAugTwo[2*jPt + 1])**2

            #for a given original point in a PD, pairwise costs w.r.t. to all the augmented points in the other PD
            #is the same and it is equal to the cost w.r.t. to that points projection on the diagonal
            elif (iPt < numPtsOne) and (jPt >= numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[2*iPt] - pdAugTwo[numPtsTwo*2 + 2*iPt])**2 + (pdAugOne[2*iPt] - pdAugTwo[numPtsTwo*2 + 2*iPt])**2 

            elif (iPt >= numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[numPtsOne*2 + 2*jPt] - pdAugTwo[2*jPt])**2 + (pdAugOne[2*numPtsOne + 2*jPt] - pdAugTwo[2*jPt])**2 

            #the last case is (iPt >= numPtsOne) and (jPt >= numPtsTwo), for which cost between projected points on the diagonal is zero

    if (solver == "scipy"): 
        srcInd, tgtInd = spopt.linear_sum_assignment(costMat)
    elif (solver == "solve_dense"):
        srcInd, tgtInd = solve_dense(costMat)  

    #if an original point is mapped to one of the projections on the diagonal,
    #then that point is mapped to its projection on the diagonal. 
    for iPt in range(numPtsOne):
        if (tgtInd[iPt] >= numPtsTwo):
            tgtInd[iPt] = numPtsTwo+iPt  

    for iPt in range(numPtsTwo):
        if (srcInd[iPt] >= numPtsOne):
            tgtInd[iPt] = numPtsTwo+iPt  

    return tgtInd

"""
Find 2-Wasserstein distance based matching, between two PDs for a given homology dimension.
The PDs are assumed to have equal number of points for that homology dimension and the PDs are not augmented by diagonal projection of points from the other PD.
Meant to be used with 0-homology. 
""" 
def findMatchConstrained(pdOne, pdTwo, homDim):

    pdOnePick = [x for x in pdOne if (x[0] == homDim)] #Pick only points of given homology dimension
    pdTwoPick = [x for x in pdTwo if (x[0] == homDim)] #from the persistence diagrams.

    if (len(pdOnePick) != len(pdTwoPick)):
        print("Error in findMatchConstrained: The two persistence diagrams should have the same number of points for given homology dimension. Instead, they are %d and %d." % (len(pdOnePick),len(pdTwoPick)))
        return None

    #the main aspect is about defining the cost matrix
    numPts = len(pdOnePick)
    costMat = np.zeros((numPts, numPts),dtype=np.float64)

    for iPt in range(numPts):
        for jPt in range(numPts):
            #pairwise costs between original points
            costMat[iPt,jPt] = (pdOnePick[iPt][1][0] - pdTwoPick[jPt][1][0])**2 + (pdOnePick[iPt][1][1] - pdTwoPick[jPt][1][1])**2 

#    srcInd, tgtInd = spopt.linear_sum_assignment(costMat)
    srcInd, tgtInd = solve_dense(costMat) 

    return tgtInd

"""
Find 2-Wasserstein distance based matching, between two PDs of a given homology dimension.
The PDs are assumed to have equal number of points for that homology dimension and the PDs are not augmented by diagonal projection of points from the other PD.
Meant to be used with 0-homology. 
""" 
def findMatchConstrainedTensor(pdOne, pdTwo):

    if (len(pdOne) != len(pdTwo)):
        print("Error in findMatchConstrained: The two persistence diagrams should have the same number of points for given homology dimension. Instead, they are %d and %d." % (len(pdOne),len(pdTwo)))
        return None

    #the main aspect is about defining the cost matrix
    numPts = int(0.5*len(pdOne))
    costMat = np.zeros((numPts, numPts),dtype=np.float64)

    for iPt in range(numPts):
        for jPt in range(numPts):
            #pairwise costs between original points
            costMat[iPt,jPt] = (pdOne[2*iPt] - pdTwo[2*jPt])**2 + (pdOne[2*iPt + 1] - pdTwo[2*jPt + 1])**2 

#    srcInd, tgtInd = spopt.linear_sum_assignment(costMat)
    srcInd, tgtInd = solve_dense(costMat) 

    return tgtInd

def attachEdgeFilt(pcPts, ptIdx, filtration):
    #find attaching edge of a given simplex, based on filtration value.
    #the edge with the maximum length (for Rips filtration).

    pcDim = pcPts.shape[1]

    pcSelect = np.empty((0,pcDim), dtype = np.float64)

    for iPt in ptIdx:
        pcSelect = np.append(pcSelect, np.reshape(np.asarray(pcPts[iPt,:]),(1,pcDim)), axis = 0)
 
    (edgeInd0, edgeInd1) = (None, None)

    numPts = len(ptIdx)

    edgeList = [(ind0,ind1) for ind0 in range(numPts) for ind1 in range(numPts) if ind0 < ind1]

    maxLen = 0

    for (ind0,ind1) in edgeList:
        length = filtration((ptIdx[ind0], ptIdx[ind1]))

        if length > maxLen:
            maxLen = length
            (edgeInd0, edgeInd1) = (ptIdx[ind0], ptIdx[ind1])

    return (edgeInd0, edgeInd1) 

def attachEdgeFilt(pcPts, ptIdx, filtration):
    #find attaching edge of a given simplex, based on filtration value.
    #the edge with the maximum length (for Rips filtration).

    pcDim = pcPts.shape[1]

    pcSelect = np.empty((0,pcDim), dtype = np.float64)

    for iPt in ptIdx:
        pcSelect = np.append(pcSelect, np.reshape(np.asarray(pcPts[iPt,:]),(1,pcDim)), axis = 0)
 
    (edgeInd0, edgeInd1) = (None, None)

    numPts = len(ptIdx)

    edgeList = [(ind0,ind1) for ind0 in range(numPts) for ind1 in range(numPts) if ind0 < ind1]

    maxLen = 0

    for (ind0,ind1) in edgeList:
        length = filtration((ptIdx[ind0], ptIdx[ind1]))

        if length > maxLen:
            maxLen = length
            (edgeInd0, edgeInd1) = (ptIdx[ind0], ptIdx[ind1])

    return (edgeInd0, edgeInd1) 

def attachEdge(pcPts, ptIdx):
    #find attaching edge of a given simplex.
    #the edge with the maximum length (for Rips filtration).

    pcDim = pcPts.shape[1]

    pcSelect = np.empty((0,pcDim), dtype = np.float64)

    for iPt in ptIdx:
        pcSelect = np.append(pcSelect, np.reshape(np.asarray(pcPts[iPt,:]),(1,pcDim)), axis = 0)
 
    (edgeInd0, edgeInd1) = (None, None)

    numPts = len(ptIdx)

    edgeList = [(ind0,ind1) for ind0 in range(numPts) for ind1 in range(numPts) if ind0 < ind1]

    maxLen = 0

    for (ind0,ind1) in edgeList:
        length = np.linalg.norm(np.array(pcSelect[ind0]) - np.array(pcSelect[ind1]))

        if length > maxLen:
            maxLen = length
            (edgeInd0, edgeInd1) = (ptIdx[ind0], ptIdx[ind1])

    return (edgeInd0, edgeInd1)
 
def attachEdgePairDist(pairDist, ptIdx):
    #find attaching edge of a given simplex.
    #the edge with the maximum length (for Rips filtration).
    #pairDist: flat vector of pairwise distances.

    numPairs = len(pairDist)

    numPts = int((1 + np.sqrt(1 + 8*numPairs))/2) 

    simSiz = len(ptIdx)

    edgeList = [(ind0,ind1) for ind0 in ptIdx for ind1 in ptIdx if ind0 > ind1]

    #print('edgeList')
    #print(edgeList)

    (edgeInd0, edgeInd1) = (None, None)

    maxLen = 0

    for (ind0,ind1) in edgeList:
        #print('(ind0,ind1) (%d,%d)' % (ind0,ind1))

        length = pairDist[int(ind0*(ind0 - 1)/2 + ind1)]

        if length > maxLen:
            maxLen = length
            (edgeInd0, edgeInd1) = (ind0, ind1)
    
    return (edgeInd0, edgeInd1)

"""
Computation of information required for the chain rule that is between the point cloud representation and PD representation.
The persistence diagram is obtained through DTM.
"""
def buildDtmPcPdInfo(pcPts, homDim, maxEdgeLen, kNN, timerFlag = False):

    #If timerFlag is enabled, then measure and print timing information.
    if (timerFlag):
        startTime = time.clock()

    numHomDim = len(homDim)

    DTM_values,NN_Dist,NN = DTM_revised(pcPts,pcPts,kNN)
    distances = EuclideanDistances(pcPts)

    if (timerFlag):
        print('buildDtmPcPdInfo: computing DTM weights took %f seconds' % (time.clock() - startTime))
        startTime = time.clock()

    simplex_tree = StructureW(pcPts,DTM_values,distances,edge_max=maxEdgeLen)

    if (timerFlag):
        print('buildDtmPcPdInfo: building the simplex tree data structure took %f seconds' % (time.clock() - startTime)) 
        startTime = time.clock()

    filtrationFunc = simplex_tree.filtration

    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

    if (timerFlag):
        print('buildDtmPcPdInfo: computing persistence homology took %f' % (time.clock() - startTime))

    persistencePairs = simplex_tree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
    #note this array is not alligned with the array of (birth,death) pairs computed by persistence 

    filtPairs = np.empty((0,2), dtype = np.float64)

    srcPD = [] 

    for iHomDim in homDim: 
        iCnt = 0

        for iDiag in diag:
            curFiltPair = np.array([[simplex_tree.filtration(persistencePairs[iCnt][0]), simplex_tree.filtration(persistencePairs[iCnt][1])]])

            if (np.isinf(curFiltPair[0,1])):
                curFiltPair[0,1] = maxEdgeLen

            filtPairs = np.append(filtPairs, curFiltPair, axis = 0)

            birthTime = iDiag[1][0];
            deathTime = iDiag[1][1]; 

            if (iDiag[0] == iHomDim):
                if (np.isinf(deathTime)):
                    deathTime = maxEdgeLen

                srcPD.append((iHomDim,(birthTime, deathTime)))

            iCnt = iCnt + 1 

    #find the simplex corresponding to a given birth or death by comparing filtPairs and srcPD.

    numPts = len(pcPts)

    pcDim = len(pcPts[0])

    pcPDInfo = [[[] for iPt in range(numPts)] for i in range(numHomDim)]

    homCnt = 0

    for iHomDim in homDim:
        srcPdPick = [x for x in srcPD if (x[0] == iHomDim)] 

        iPersPair = 0

        pdProcess = []

        for iFilt in filtPairs:
            iPD = 0
            noMatchPt = True
            while (noMatchPt) and (iPD < len(srcPdPick)):
                if (iFilt[0] == srcPdPick[iPD][1][0]) and (iFilt[1] == srcPdPick[iPD][1][1]) and (iPD not in pdProcess):
                    noMatchPt = False
                    pdProcess.append(iPD) 
                else:
                    iPD = iPD + 1

            if not noMatchPt:
            #for each point in the PD, there is a birth simplex and a death simplex.
                for iSimplex in range(2):
                    if (len(persistencePairs[iPersPair][iSimplex]) > 1):
                        (maxInd0, maxInd1) = attachEdgeFilt(pcPts, persistencePairs[iPersPair][iSimplex],filtrationFunc)

                        attachLen = np.linalg.norm(pcPts[maxInd0,:] - pcPts[maxInd1,:])

                        derListZero = [(pcPts[maxInd0,iDim]-pcPts[maxInd1,iDim])/attachLen for iDim in range(pcDim)] #compute derivative of birth or death time (i.e., attachLen)
                        derListOne = [(pcPts[maxInd1,iDim]-pcPts[maxInd0,iDim])/attachLen for iDim in range(pcDim)]  #w.r.t. each coordinate of the point in the PC.

                        derListKNNZero = [0]*pcDim
                        derListKNNOne = [0]*pcDim

                        for iNN in range(kNN):
                            derListKNNZero = [sum(x) for x in zip(derListKNNZero,[(pcPts[maxInd0,iDim]-pcPts[NN[maxInd0,iNN],iDim]) for iDim in range(pcDim)])]
                            derListKNNOne = [sum(x) for x in zip(derListKNNOne,[(pcPts[maxInd1,iDim]-pcPts[NN[maxInd1,iNN],iDim]) for iDim in range(pcDim)])]

                        derListKNNZero = [(1./(kNN*DTM_values[maxInd0]))*x for x in derListKNNZero]
                        derListKNNOne = [(1./(kNN*DTM_values[maxInd1]))*x for x in derListKNNOne]

                        pcPDInfo[homCnt][maxInd0].append((iPD,iSimplex,np.array([0.5*sum(x) for x in zip(derListZero,derListKNNZero)]).reshape((pcDim,1))))
                                                                                                                   #for a given point in the PC, associate a point in the PD,
                        pcPDInfo[homCnt][maxInd1].append((iPD,iSimplex,np.array([0.5*sum(x) for x in zip(derListOne,derListKNNOne)]).reshape((pcDim,1))))   #whether it was a birth or death simplex  
                                                                                                                   #and derivatives w.r.t. all the coordinates of the point in the PC.
                                                                                                                   #this is done for both the points in the attaching edge of the simplex. 
                        for iNN in range(kNN):
                            derListKNNZero = np.array([(0.5/(kNN*DTM_values[maxInd0]))*(pcPts[NN[maxInd0,iNN],iDim]-pcPts[maxInd0,iDim]) for iDim in range(pcDim)]).reshape((pcDim,1))
                            derListKNNOne = np.array([(0.5/(kNN*DTM_values[maxInd1]))*(pcPts[NN[maxInd1,iNN],iDim]-pcPts[maxInd1,iDim]) for iDim in range(pcDim)]).reshape((pcDim,1))
 
                            pcPDInfo[homCnt][NN[maxInd0,iNN]].append((iPD,iSimplex,derListKNNZero))
                            pcPDInfo[homCnt][NN[maxInd1,iNN]].append((iPD,iSimplex,derListKNNOne))

                    elif (len(persistencePairs[iPersPair][iSimplex]) == 1): #a 0-simplex, it is a point and it will always be a birth simplex.
                        if iSimplex:
                            print('buildDtmPcPdInfo:SOMETHING WRONG!')
                            print(persistencePairs[iPersPair][iSimplex])
     
                        ptInd = persistencePairs[iPersPair][iSimplex][0]
 
                        derList = [0]*pcDim

                        for iNN in range(kNN):
                            derList = [sum(x) for x in zip(derList,[(pcPts[ptInd,iDim]-pcPts[NN[ptInd,iNN],iDim]) for iDim in range(pcDim)])]

                        derList = np.array([(1./(kNN*DTM_values[ptInd]))*x for x in derList]).reshape((pcDim,1))

                        pcPDInfo[homCnt][ptInd].append((iPD,0,derList)) #a point will always be a birth simplex. hence, second parameter is 0.

                        for iNN in range(kNN):
                            derList = np.array([(1./(kNN*DTM_values[ptInd]))*(pcPts[NN[ptInd,iNN],iDim]-pcPts[ptInd,iDim]) for iDim in range(pcDim)]).reshape((pcDim,1))

                            pcPDInfo[homCnt][NN[ptInd,iNN]].append((iPD,0,derList)) #a point will always be a birth simplex. hence, second parameter is 0.
 
            iPersPair = iPersPair + 1 

        homCnt = homCnt + 1

    return srcPD, pcPDInfo 

"""
Computation of information required for the chain rule that is between the point cloud representation and PD representation.
The persistence diagram is obtained through Rips filtration.
"""
def buildRipsPcPdInfo(pcPts, homDim, maxEdgeLen, timerFlag = False):

    #If timerFlag is enabled, then measure and print timing information.
    if (timerFlag):
        startTime = time.clock()

    numHomDim = len(homDim)

    maxDimSimplex = max(homDim) + 1 #considering only sufficient homology dimension 

    rips_complex = gd.RipsComplex(pcPts, max_edge_length=maxEdgeLen)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxDimSimplex)
 
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

    if (timerFlag):
        print('buildRipsPcPdInfo: computing persistence homology took %f' % (time.clock() - startTime))

    persistencePairs = simplex_tree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
                                                        #note this array is not alligned with the array of (birth,death) pairs computed by persistence 

    print(persistencePairs)   
 
    filtPairs = np.empty((0,2), dtype = np.float64)

    srcPD = [] 

    for iHomDim in homDim: 
        iCnt = 0

        for iDiag in diag:
            curFiltPair = np.array([[simplex_tree.filtration(persistencePairs[iCnt][0]), simplex_tree.filtration(persistencePairs[iCnt][1])]])

            if (np.isinf(curFiltPair[0,1])):
                curFiltPair[0,1] = maxEdgeLen

            filtPairs = np.append(filtPairs, curFiltPair, axis = 0)

            birthTime = iDiag[1][0];
            deathTime = iDiag[1][1]; 

            if (iDiag[0] == iHomDim):
                if (np.isinf(deathTime)):
                    deathTime = maxEdgeLen

                srcPD.append((iHomDim,(birthTime, deathTime)))

            iCnt = iCnt + 1 

    #find the simplex corresponding to a given birth or death by comparing filtPairs and srcPD.

    numPts = len(pcPts)

    pcDim = len(pcPts[0])

    pcPDInfo = [[[] for iPt in range(numPts)] for i in range(numHomDim)]

    homCnt = 0

    for iHomDim in homDim:
        srcPdPick = [x for x in srcPD if (x[0] == iHomDim)] 

        iPersPair = 0

        pdProcess = []

        for iFilt in filtPairs:
            iPD = 0
            noMatchPt = True
            while (noMatchPt) and (iPD < len(srcPdPick)):
                if (iFilt[0] == srcPdPick[iPD][1][0]) and (iFilt[1] == srcPdPick[iPD][1][1]) and (iPD not in pdProcess):
                    noMatchPt = False
                    pdProcess.append(iPD) 
                else:
                    iPD = iPD + 1

            if not noMatchPt: #if a match has been found
            #for each point in the PD, there is a birth simplex and a death simplex.
                for iSimplex in range(2):
                    if (len(persistencePairs[iPersPair][iSimplex]) > 1): #consider only edges and above.
                        (maxInd0, maxInd1) = attachEdge(pcPts, persistencePairs[iPersPair][iSimplex])

                        attachLen = 0

                        for iDim in range(pcDim):
                            attachLen = attachLen + (pcPts[maxInd0,iDim] - pcPts[maxInd1,iDim])**2

                        attachLen = np.sqrt(attachLen)

                        derListZero = []
                        derListOne = []

                        for iDim in range(pcDim):
                            derListZero.append((pcPts[maxInd0,iDim]-pcPts[maxInd1,iDim])/attachLen) #compute derivative of birth or death time (i.e., attachLen)
                            derListOne.append((pcPts[maxInd1,iDim]-pcPts[maxInd0,iDim])/attachLen)  #w.r.t. each coordinate of the point in the PC.

                        pcPDInfo[homCnt][maxInd0].append((iPD,iSimplex,np.array(derListZero).reshape((pcDim,1))))
                        pcPDInfo[homCnt][maxInd1].append((iPD,iSimplex,np.array(derListOne).reshape((pcDim,1))))
                        #for a given point in the PC, associate a point in the PD, whether it was a birth or death simplex 
                        #and list of derivatives w.r.t. all the coordinates of the point in the PC.
                        #this is done for both the points in the attaching edge of the simplex. 
            iPersPair = iPersPair + 1 #for iSimplex 

        homCnt = homCnt + 1

    return srcPD, pcPDInfo 

"""
Create time delay embedding (TDE) based point cloud from time series data.
"""
def convertTde(timeSeries, skip=1, delay=1, dimension=3):
    X = []

    for j in range((len(timeSeries) - (dimension - 1) * delay) // skip):
        pt = []
        for k in range(dimension):
            coord = timeSeries[j * skip + k * delay]
            pt.append(coord)
        X.append(pt)
    X = np.array(X)

    #print(X.shape)

    return X

"""
Return greatest power of 2 less than given number
"""
def findGreatestPowOf2(ipNum):

    for iNum in range(ipNum, 0, -1):
        if (iNum & (iNum - 1) == 0):
            return iNum

def gudhiToTensorList(tgtPDGudhi, homDim, maxEdgeLen):
 
    diagList = [[] for i in range(len(homDim))]

    diagTensorList = [] 

    for iPDPt in tgtPDGudhi:
        iHomDim = homDim.index(iPDPt[0])  
        diagList[iHomDim].append(iPDPt[1][0]) #append the birth time
        
        deathTime = iPDPt[1][1]

        if np.isinf(deathTime):
            deathTime = maxEdgeLen 

        diagList[iHomDim].append(deathTime) #append the death time

    for iHomDim in range(len(homDim)):
        diagTensorList.append(torch.tensor(diagList[iHomDim]))

    return diagTensorList
