from scipy import optimize as spopt
from lapsolver import solve_dense
import numpy as np
import gudhi as gd
import csv
import pandas as pd
import time
import sys

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
    return X

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
Obtain persistence diagram (Rips based) from a given time series.
"""
def pdFromTimeSeries(tgtTimeSeries, pcDim, homDim, maxEdgeLen, tdeSkip = 1, tdeDelay = 1):
    #homDim indicates which homology dimension to consider.

    pcPts = convertTde(tgtTimeSeries, skip = tdeSkip, delay = tdeDelay, dimension = pcDim) #create point cloud (PC) for time series 

    maxDimSimplex = homDim + 1 #considering only sufficient homology dimension 

    rips_complex = gd.RipsComplex(pcPts, max_edge_length=maxEdgeLen)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxDimSimplex)

    #PD of the point cloud
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
 
    tgtPD = [] #target PD is a list of tuples, same as Gudhi format.

    maxPers = 0
    maxInd = 0

    for iDiag in diag:
        if (iDiag[0] == homDim):
            birthTime = iDiag[1][0]
            deathTime = iDiag[1][1] 

            if (np.isinf(deathTime)): #replace inf value with max. edge length
                deathTime = maxEdgeLen

            tgtPD.append((iDiag[0],(birthTime,deathTime)))

    return tgtPD

"""
Find 2-Wasserstein distance based matching, between two PDs for a given homology dimension.
The set of points for each PD is augmented with diagonal projection of points from the other PD. 
""" 
def findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, solver="scipy"):
#pdAugOne and pdAugTwo are persistence diagrams, which also contain projections on the diagonal
#of points from the other PD, for the given homology dimension. They are usually created by the function augmentPd.  

    #numPtsOne is the number of points in pdOne for given homology dimension before augmentation.
    #numPtsTwo is the number of points in pdTwo for given homology dimension before augmentation.
    totPts = numPtsOne + numPtsTwo

    #the main aspect is about defining the cost matrix

    costMat = np.zeros((totPts, totPts),dtype=np.float64)

    for iPt in range(totPts):
        for jPt in range(totPts):
            #pairwise costs between original (not augmented) points 
            if (iPt < numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[iPt][1][0] - pdAugTwo[jPt][1][0])**2 + (pdAugOne[iPt][1][1] - pdAugTwo[jPt][1][1])**2

            #for a given original point in a PD, pairwise costs w.r.t. to all the augmented points in the other PD
            #is the same and it is equal to the cost w.r.t. to that points projection on the diagonal
            elif (iPt < numPtsOne) and (jPt >= numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[iPt][1][0] - pdAugTwo[numPtsTwo + iPt][1][0])**2 + (pdAugOne[iPt][1][1] - pdAugTwo[numPtsTwo + iPt][1][1])**2 

            elif (iPt >= numPtsOne) and (jPt < numPtsTwo):
                costMat[iPt,jPt] = (pdAugOne[numPtsOne + jPt][1][0] - pdAugTwo[jPt][1][0])**2 + (pdAugOne[numPtsOne + jPt][1][1] - pdAugTwo[jPt][1][1])**2 

            #the last case is (iPt >= numPtsOne) and (jPt >= numPtsTwo), for which cost between projected points on the diagonal is zero

    startTime = time.clock()
    if (solver == "scipy"): 
        srcInd, tgtInd = spopt.linear_sum_assignment(costMat)
    elif (solver == "solve_dense"):
        srcInd, tgtInd = solve_dense(costMat)
    print("Matching took %f seconds." % (time.clock() - startTime)) 

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
Compute 2-Wasserstein distance between two PDs given the matching. 
""" 
def comp2WassCost(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, matchPts):

    funcVal = 0

    for iPtPD in range(numPtsOne):
        funcVal += (pdAugOne[iPtPD][1][0] - pdAugTwo[matchPts[iPtPD]][1][0])**2 + (pdAugOne[iPtPD][1][1] - pdAugTwo[matchPts[iPtPD]][1][1])**2

    numPtsTotal = numPtsOne + numPtsTwo

    #taking care of points in pdTwo that are matched to their projection on the diagonal
    for iPtPd in range(numPtsOne,numPtsTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
        if (matchPts[iPtPd] < numPtsTwo):
            diagVal = (pdAugTwo[matchPts[iPtPd]][1][0] + pdAugTwo[matchPts[iPtPd]][1][1])/2
            funcVal += (pdAugTwo[matchPts[iPtPd]][1][0] - diagVal)**2 + (pdAugTwo[matchPts[iPtPd]][1][1] - diagVal)**2

    return np.sqrt(funcVal) 

if (len(sys.argv) < 2):
    print("Pass name of matching algorithm as argument: scipy or solve_dense")
    sys.exit() 

#Some parameter values have to be set
pcDim = 3 #dimensionality of the point cloud (PC) space.
maxEdgeLen = 10 #saturation time for the filtration.

homDim = 0 #homology dimension for which you want to compute the match.

#EEG data
eeg_data = np.array(pd.read_csv("eeg_train.csv",header=None))

timeSeriesOne = 0.01*eeg_data[304,:]
timeSeriesTwo = 0.01*eeg_data[569,:]

pdOne = pdFromTimeSeries(timeSeriesOne,pcDim,homDim,maxEdgeLen)
pdTwo = pdFromTimeSeries(timeSeriesTwo,pcDim,homDim,maxEdgeLen)

pdAugOne, pdAugTwo, numPtsOne, numPtsTwo = augmentPd(pdOne, pdTwo, homDim) #augment each PD with diagonal projections from other PD. 
#numPtsOne is the number of points in pdOne for given homology dimension before augmentation.
#numPtsTwo is the number of points in pdTwo for given homology dimension before augmentation.

#last option is either "scipy" or "solve_dense"
matchPts = findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, sys.argv[1])

print("2-Wasserstein distance is %f." % (comp2WassCost(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, matchPts)))
