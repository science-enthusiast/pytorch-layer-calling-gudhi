"""
Created on 31/01/2019

@author Hariprasad KANNAN - Inria DataShape

All rights reserved
"""
 
#continuation for time series
#this code works with time series representation.
#DTM based filtration

from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as spopt
import scipy.interpolate as interp
from random import gauss
import scipy.fftpack as fftpack
import numpy as np
import matplotlib.pyplot as plt
import time
from common import *

"""
Return function and gradien values for a given value of time series.
Filtration computed based on DTM.
Function value is the squared 2-Wasserstein distance between the PD
of current time series and the target PD.
"""
def compFuncGradDtm(timeSeriesData, pcDim, pdTwo, homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1):
    #return function and gradient values
    #timeSeriesData: a given time series data
    #pcDim: dimension of the TDE space.
    #pdTwo: target PD (remains fixed).
    #homDim: homology dimensions over which the cost is computed.
    #strictDim: homology dimensions over which strict one-to-ne match between points is enforced. 
    #maxEdgeLen: maximum edge length of the Rips filtration.
    #kNN: number of nearest neighbours for DTM filtration.

    pcPts = convertTde(timeSeriesData, skip = tdeSkip, delay = tdeDelay, dimension = pcDim)

    pcSiz = pcPts.shape[0]
    pcDim = pcPts.shape[1]

    tsSiz = len(timeSeriesData)

    #need to associate a given coordinate in the point cloud to points on the PD.
    #pcPdInfoAll: carries this information for all points in the point cloud.
    #pdOne: PD of the current point cloud. 
    pdOne, pcPdInfoAll = buildDtmPcPdInfo(pcPts, homDim, maxEdgeLen, kNN)

    homCnt = 0

    funcVal = 0

    gradVec = np.zeros((tsSiz,1),dtype=np.float64)

    for iHomDim in homDim:
        if iHomDim in strictDim: #enforce strict one-to-one match. Note for that homology dimension, the two PDs should have same number of points. 
            pdAugOne = list(pdOne)
            pdAugTwo = list(pdTwo)
            #print("length of pdAugOne %d pdAugTwo %d" % (len(pdAugOne), len(pdAugTwo)))
            matchPts = findMatchConstrained(pdOne, pdTwo, iHomDim) #function to match equal number of points. 
        else:
            pdAugOne, pdAugTwo, numPtsOne, numPtsTwo = augmentPd(pdOne, pdTwo, iHomDim) #augment PDs with diagonal projections from other PD. 
            matchPts = findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, iHomDim, "solve_dense")
            #numPtsOne and numPtsTwo are the number of points for given homology dimension in the original persistence diagrams.
            #It is returned by augmentPd and needed by findMatch.

        pdOnePick = [x for x in pdOne if (x[0] == iHomDim)]
        pdAugTwoPick = [x for x in pdAugTwo if (x[0] == iHomDim)] 

        #numPtsOne and numPtsTwo are the number of points for given homology dimension
        #in the original persistence diagrams. Only numPtsOne is needed in the following lines.
        numPtsOne = len(pdOnePick)

        for iPtPD in range(numPtsOne):
            funcVal = funcVal + (pdOnePick[iPtPD][1][0] - pdAugTwoPick[matchPts[iPtPD]][1][0])**2 + (pdOnePick[iPtPD][1][1] - pdAugTwoPick[matchPts[iPtPD]][1][1])**2

        numPtsOneTotal = len(pdAugTwoPick)

        #taking care of points in pdTwo that are matched to their projection on the diagonal
        for iPtPd in range(numPtsOne,numPtsOneTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
            if (matchPts[iPtPd] < numPtsTwo):
                diagVal = (pdAugTwoPick[matchPts[iPtPd]][1][0] + pdAugTwoPick[matchPts[iPtPd]][1][1])/2
                funcVal = funcVal + (pdAugTwoPick[matchPts[iPtPd]][1][0] - diagVal)**2 + (pdAugTwoPick[matchPts[iPtPd]][1][1] - diagVal)**2

        gradVecPC = np.zeros((pcDim*pcSiz,1),dtype=np.float64)

        pcPdInfoCur = pcPdInfoAll[homCnt]

        for iPtPC in range(pcSiz): 
            for iPCPD in pcPdInfoCur[iPtPC]:
                iPD = iPCPD[0] #point in the PD associated with the current point in the PC.
                bdFlag = iPCPD[1] #whether the point in the PC is associated with a birth or a death simplex of the point in the PD.
                edgeDer = iPCPD[2] #derivative of the birth or death time w.r.t. coordinate of the point in the PC.

                partDer = 2*(pdOnePick[iPD][1][bdFlag] - pdAugTwoPick[matchPts[iPD]][1][bdFlag])

                gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] = gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] + partDer*edgeDer

        '''
        lowLim = pcDim-2

        upLim = tsSiz-(pcDim-1)*tdeDelay 

        for j in range(tsSiz):
            if (j > lowLim) and (j < upLim):
                for iDim in range(pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j >= upLim):
                for iDim in range(pcDim-tsSiz+j,pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j <= lowLim):
                for iDim in range(j):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]
        '''

        for iPt in range(pcSiz):
            for iDim in range(pcDim):
                gradVec[iPt*tdeSkip + iDim*tdeDelay] += gradVecPC[iPt*pcDim + iDim] 

        homCnt = homCnt + 1 
 
    return funcVal, gradVec

"""
Return function and gradien values for a given value of time series.
Filtration computed based on Rips.
Function value is the 2-Wasserstein distance between the PD of current time series and the target PD.
"""
def compFuncGradRips(timeSeriesData, pcDim, pdTwo, homDim, strictDim, maxEdgeLen):
    #return function and gradient values
    #timeSeriesData: a given time series data
    #pcDim: dimension of the TDE space.
    #pdTwo: target PD (remains fixed).
    #homDim: Homology dimensions over which the cost is computed.
    #strictDim: Homology dimensions over which strict one-to-one match is enforced. 
    #maxEdgeLen: maximum edge length of the Rips filtration.

    pcPts = convertTde(timeSeriesData,dimension=pcDim)

    pcSiz = pcPts.shape[0]
    pcDim = pcPts.shape[1]

    tsSiz = len(timeSeriesData)

    #need to associate a given coordinate in the point cloud to points on the PD.
    #pcPdInfoAll: carries this information for all points in the point cloud.
    #pdOne: PD of the current point cloud. 
    pdOne, pcPdInfoAll = buildRipsPcPdInfo(pcPts, homDim, maxEdgeLen)

    homCnt = 0

    funcVal = 0

    gradVec = np.zeros((tsSiz,1),dtype=np.float64)

    for iHomDim in homDim:
        if iHomDim in strictDim: #enforce strict one-to-one match. Note for that homology dimension, the two PDs should have same number of points. 
            pdAugOne = list(pdOne)
            pdAugTwo = list(pdTwo)
            matchPts = findMatchConstrained(pdOne, pdTwo, iHomDim) #function to match equal number of points. 
        else:
            pdAugOne, pdAugTwo, numPtsOne, numPtsTwo = augmentPd(pdOne, pdTwo, iHomDim) #augment PDs with diagonal projections from other PD. 
            matchPts = findMatch(pdAugOne, pdAugTwo, numPtsOne, numPtsTwo, iHomDim, "solve_dense")
            #numPtsOne and numPtsTwo are the number of points for given homology dimension in the original persistence diagrams.
            #It is returned by augmentPd and needed by findMatch.

        pdOnePick = [x for x in pdOne if (x[0] == iHomDim)]
        pdAugTwoPick = [x for x in pdAugTwo if (x[0] == iHomDim)] 

        #numPtsOne and numPtsTwo are the number of points for given homology dimension
        #in the original persistence diagrams. Only numPtsOne is needed in the following lines.
        numPtsOne = len(pdOnePick)

        for iPtPD in range(numPtsOne):
            funcVal = funcVal + (pdOnePick[iPtPD][1][0] - pdAugTwoPick[matchPts[iPtPD]][1][0])**2 + (pdOnePick[iPtPD][1][1] - pdAugTwoPick[matchPts[iPtPD]][1][1])**2

        numPtsOneTotal = len(pdAugTwoPick)

        #taking care of points in pdTwo that are matched to their projection on the diagonal
        for iPtPd in range(numPtsOne,numPtsOneTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
            if (matchPts[iPtPd] < numPtsTwo):
                diagVal = (pdAugTwoPick[matchPts[iPtPd]][1][0] + pdAugTwoPick[matchPts[iPtPd]][1][1])/2
                funcVal = funcVal + (pdAugTwoPick[matchPts[iPtPd]][1][0] - diagVal)**2 + (pdAugTwoPick[matchPts[iPtPd]][1][1] - diagVal)**2

        gradVecPC = np.zeros((pcDim*pcSiz,1),dtype=np.float64)

        pcPdInfoCur = pcPdInfoAll[homCnt]

        for iPtPC in range(pcSiz): 
            for iPCPD in pcPdInfoCur[iPtPC]:
                iPD = iPCPD[0] #point in the PD associated with the current point in the PC.
                bdFlag = iPCPD[1] #whether the point in the PC is associated with a birth or a death simplex of the point in the PD.
                edgeDer = iPCPD[2] #derivative of the birth or death time w.r.t. coordinate of the point in the PC.

                partDer = 2*(pdOnePick[iPD][1][bdFlag] - pdAugTwoPick[matchPts[iPD]][1][bdFlag])

                gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] = gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] + partDer*edgeDer

        lowLim = pcDim-2

        upLim = tsSiz-pcDim+1 

        for j in range(tsSiz):
            if (j > lowLim) and (j < upLim):
                for iDim in range(pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j >= upLim):
                for iDim in range(pcDim-tsSiz+j,pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j <= lowLim):
                for iDim in range(j):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

        homCnt = homCnt + 1 
 
    return funcVal, gradVec

"""
Return function and gradien values for a given value of time series.
Filtration computed based on DTM.
The cost is the sum of the weighted squared distance to the origin of the points
in the PD.  
"""
def compFuncGradDtmWtMov(timeSeriesData, pcDim, ptWts, homDim, maxEdgeLen, kNN):
    #return function and gradient values
    #timeSeriesData: a given time series data
    #pcDim: dimension of the TDE space.
    #ptWts: Each point in the PD is penalized by a weight.
    #maxEdgeLen: maximum edge length of the Rips filtration.
    #kNN: number of nearest neighbours for DTM filtration.

    #function value is the 2-Wasserstein distance between the PD of current time series and the target PD.

    pcPts = convertTde(timeSeriesData,dimension=pcDim)

    pcSiz = pcPts.shape[0]
    pcDim = pcPts.shape[1]

#    addNoise = np.random.normal(0,1e-6,(pcSiz,pcDim))

#    pcPts = pcPts + addNoise

    tsSiz = len(timeSeriesData)

    #need to associate a given coordinate in the point cloud to points on the PD.
    #pcPdInfoAll: carries this information for all points in the point cloud.
    #pdOne: PD of the current point cloud. 
    pdOne, pcPdInfoAll = buildDtmPcPdInfo(pcPts, homDim, maxEdgeLen, kNN)

    homCnt = 0

    funcVal = 0

    gradVec = np.zeros((tsSiz,),dtype=np.float64)

    for iHomDim in homDim:
        pdOnePick = [x for x in pdOne if (x[0] == iHomDim)]

        #numPtsOne and numPtsTwo are the number of points for given homology dimension
        #in the original persistence diagrams. Only numPtsOne is needed in the following lines.
        numPtsOne = len(pdOnePick)

        curPtWts = ptWts[iHomDim] 

        for iPtPD in range(numPtsOne):
            funcVal = funcVal + curPtWts[iPtPD]*(pdOnePick[iPtPD][1][0]**2 + pdOnePick[iPtPD][1][1]**2)

        gradVecPC = np.zeros((pcDim*pcSiz,1),dtype=np.float64)

        pcPdInfoCur = pcPdInfoAll[homCnt]

        for iPtPC in range(pcSiz): 
            for iPCPD in pcPdInfoCur[iPtPC]:
                iPD = iPCPD[0] #point in the PD associated with the current point in the PC.
                bdFlag = iPCPD[1] #whether the point in the PC is associated with a birth or a death simplex of the point in the PD.
                edgeDer = iPCPD[2] #derivative of the birth or death time w.r.t. coordinate of the point in the PC.

                partDer = 2*curPtWts[iPD]*pdOnePick[iPD][1][bdFlag]

                gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] = gradVecPC[iPtPC*pcDim:iPtPC*pcDim + pcDim] + partDer*edgeDer

        lowLim = pcDim-2

        upLim = tsSiz-pcDim+1 

        for j in range(tsSiz):
            if (j > lowLim) and (j < upLim):
                for iDim in range(pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j >= upLim):
                for iDim in range(pcDim-tsSiz+j,pcDim):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

            if (j <= lowLim):
                for iDim in range(j):
                    gradVec[j] = gradVec[j] + gradVecPC[pcDim*(j-iDim) + iDim]

        homCnt = homCnt + 1 
 
    return funcVal, gradVec

def compFuncGradPCOT(curPCFlat, tgtPC):

    pcSiz, pcDim = tgtPC.shape

    curPC = np.reshape(curPCFlat,(pcSiz,pcDim))

    tsSiz = pcSiz + pcDim - 1
 
    costMat = np.zeros((pcSiz, pcSiz), dtype=np.float64)

    for iPt in range(pcSiz):
        for jPt in range(pcSiz):
            #pairwise costs between original points
            costMat[iPt,jPt] = np.dot(curPC[iPt,:] - tgtPC[jPt,:], curPC[iPt,:] - tgtPC[jPt,:])

    startTime = time.clock()
    srcInd, tgtInd = solve_dense(costMat)
    print("compFuncGradPCOT LAP solver took %f seconds." % (time.clock() - startTime)) 

    funcVal = 0

    for iPt in range(pcSiz):
        funcVal += np.dot(curPC[iPt,:] - tgtPC[tgtInd[iPt],:], curPC[iPt,:] - tgtPC[tgtInd[iPt],:])

    gradVec = np.zeros((pcDim*pcSiz,), dtype=np.float64)

    for iPt in range(pcSiz):
        gradVec[iPt*pcDim:iPt*pcDim + pcDim] = 2*(curPC[iPt,:] - tgtPC[tgtInd[iPt],:])

    return funcVal, gradVec

"""
Return function and gradient values where the objective function has two terms:
One based on squared 2-Wasserstein distance between the persistence diagrams.
Another based on squared 2-Wasserstein distance between the point clouds. 
Filtration computed based on DTM.
"""
def compFGPHandOTDtm(timeSeriesData, pdTwo, tgtPC, homDim, strictDim, maxEdgeLen, kNN):
    #return function and gradient values
    #timeSeriesData: a given time series data
    #pcDim: dimension of the TDE space.
    #pdTwo: target PD (remains fixed).
    #tgtPC: target point cloud.
    #homDim: homology dimensions over which the cost is computed.
    #strictDim: homology dimensions over which strict one-to-ne match between points is enforced. 
    #maxEdgeLen: maximum edge length of the Rips filtration.
    #kNN: number of nearest neighbours for DTM filtration.

    funcVal = 0

    gradVec = np.zeros((tsSiz,1))

    pcSiz = tgtPC.shape[0]
    pcDim = tgtPC.shape[1]

    pcPts = convertTde(timeSeriesData,dimension=pcDim)

    tsSiz = len(timeSeriesData)

    phFunc, phGrad = compFuncGradDtm(timeSeriesData, pcDim, pdTwo, homDim, strictDim, maxEdgeLen, kNN)

    funcVal += phFunc

    gradVec += phGrad 

    startTime = time.clock()

    pcPtsFlat = np.reshape(pcPts,(pcDim*pcSiz,))

    otFunc, otGrad = compFuncGradPCOT(pcPtsFlat, tgtPC)
    print("Function and gradient for point cloud optimal transport cost took %f seconds" % (time.clock() - startTime))

    #gradient returned by OT routine is w.r.t. the points in the point cloud.
    #need to aggregate them with respect to the time series variables. 

    otGradTS = np.zeros((tsSiz,1),dtype=np.float64)

    lowLim = pcDim-2

    upLim = tsSiz-pcDim+1 

    for j in range(tsSiz):
        if (j > lowLim) and (j < upLim):
            for iDim in range(pcDim):
                otGradTS[j] += otGrad[pcDim*(j-iDim) + iDim]

        if (j >= upLim):
            for iDim in range(pcDim-tsSiz+j,pcDim):
                otGradTS[j] += otGrad[pcDim*(j-iDim) + iDim]

        if (j <= lowLim):
            for iDim in range(j):
                otGradTS[j] += otGrad[pcDim*(j-iDim) + iDim]

    funcVal += 0.5*otFunc
    gradVec += 0.5*otGradTS 
 
    return funcVal, gradVec

"""
Return function and gradient of a function which is the sum of the squared differences
between the top Fourier coefficients of current time series and target time series.
The gradient is computed w.r.t. the time series samples.
Note: We assume that the current time series is at least as long as the target time series.
      It is better if they are of the same length. 
Note: We assume that the current and target time series are of odd length.
      If they are of even length, then the last coefficient (as computed by fftpack.rfft) is ignored. 
"""
def compFGDftCoeff(timeSeriesData, tgtDftCoeff):

    tsSiz = len(timeSeriesData)

    numDftCoeff = int((len(tgtDftCoeff) + 1)/2) #based on the assumption that the current and the target time series are of odd length. 

    curDftCoeff = fftpack.rfft(timeSeriesData) #DFT of current signal

#    print('Target DFT coefficients')
#    print(tgtDftCoeff)

#    print('Current DFT coefficients')
#    print(curDftCoeff)

    funcVal = (curDftCoeff[0] - tgtDftCoeff[0])**2
     
    for iCoeff in range(1,numDftCoeff):
        funcVal += (curDftCoeff[2*iCoeff - 1] - tgtDftCoeff[2*iCoeff - 1])**2 + (curDftCoeff[2*iCoeff] - tgtDftCoeff[2*iCoeff])**2

    gradVec = np.zeros((tsSiz,1), dtype=np.float64)

    for iPt in range(tsSiz):
        gradVec[iPt] = 2*(curDftCoeff[0] - tgtDftCoeff[0])
        for iCoeff in range(1,numDftCoeff):
            gradVec[iPt] += 2*(curDftCoeff[2*iCoeff - 1] - tgtDftCoeff[2*iCoeff - 1])*np.cos(2*np.pi*iCoeff*iPt/tsSiz) - 2*(curDftCoeff[2*iCoeff] - tgtDftCoeff[2*iCoeff])*np.sin(2*np.pi*iCoeff*iPt/tsSiz)

    return funcVal, gradVec

"""
Return function and gradient values where the objective function has two terms:
One based on squared 2-Wasserstein distance between the persistence diagrams.
Another based on sum of squared differences between top DFT coefficients. 
Filtration computed based on DTM.
"""
def compFGPHDtmAndDftDiff(timeSeriesData, pcDim, tgtDftCoeff, pdTwo, homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1):
    #return function and gradient values
    #timeSeriesData: a given time series data
    #pcDim: dimension of the TDE space.
    #pdTwo: target PD (remains fixed).
    #tgtPC: target point cloud.
    #homDim: homology dimensions over which the cost is computed.
    #strictDim: homology dimensions over which strict one-to-ne match between points is enforced. 
    #maxEdgeLen: maximum edge length of the Rips filtration.
    #kNN: number of nearest neighbours for DTM filtration.

    funcVal = 0

    tsSiz = len(timeSeriesData)

    gradVec = np.zeros((tsSiz,1))

    pcPts = convertTde(timeSeriesData, skip = tdeSkip, delay = tdeDelay, dimension = pcDim)

    phFunc, phGrad = compFuncGradDtm(timeSeriesData, pcDim, pdTwo, homDim, strictDim, maxEdgeLen, kNN, tdeSkip = tdeSkip, tdeDelay = tdeDelay)

    print('gradient of squared 2-Wasserstein distance has shape')
    print(phGrad.shape)

    funcVal += phFunc
    gradVec += phGrad 

    startTime = time.clock()

    dftFunc, dftGrad = compFGDftCoeff(timeSeriesData, tgtDftCoeff)

    print("Function and gradient for point cloud optimal transport cost took %f seconds" % (time.clock() - startTime))

    funcVal += 0*dftFunc
    gradVec += 0*dftGrad 
 
    return funcVal, gradVec

def compFGPHMultiLevel(timeSeriesData, pcDim, dftLevels, dftTsDerMat, tsDftDerMat, pdTwoSet, homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1):

    tsSiz = len(timeSeriesData)

    numLevels = len(dftLevels)

    allDftCoeff = fftpack.rfft(timeSeriesData)

    funcVal = 0

    gradVec = np.zeros((tsSiz,1)) 

    for iLevel in range(numLevels):
        numCoeff = 1 + 2*(dftLevels[iLevel]-1)
        curDftCoeff = allDftCoeff[:numCoeff]
        curTimeSeries = fftpack.irfft(curDftCoeff,tsSiz)
        curFuncVal, curGradVec = compFuncGradDtm(curTimeSeries, pcDim, pdTwoSet[iLevel], homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1)

        curDftTsDerMat = dftTsDerMat[:numCoeff]
        curTsDftDerMat = tsDftDerMat[:,:numCoeff]

        gradVecDft = np.dot(curDftTsDerMat,curGradVec)

        gradVec += np.dot(curTsDftDerMat, gradVecDft) 

        funcVal += curFuncVal

    return funcVal, gradVec

def compFGPHMSHaar(timeSeriesData, pcDim, scaleList, haarDerMatList, pdTwoList, homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1):

    tsSiz = len(timeSeriesData) #At an earlier stage, it has been made a power of 2.

    numScales = len(scaleList)

    funcVal = 0

    gradVec = np.zeros((tsSiz,1)) 

    scaleListCnt = 0

    for iScale in scaleList:
        levSiz = int(tsSiz/iScale) #size of time series at this level. For Haar wavelets, it is also the number of wavelets.  

        curTimeSeries = np.zeros(levSiz)

        for ind in range(levSiz):
            for iSamp in range(iScale):
                curTimeSeries[ind] += timeSeriesData[ind*iScale + iSamp]

            curTimeSeries[ind] /= iScale 

#        curTimeSeries_interp = interp.interp1d(np.arange(curTimeSeries.size), curTimeSeries)

#        curTimeSeries = curTimeSeries_interp(np.linspace(0, curTimeSeries.size-1, tsSiz))

        curFuncVal, curGradVec = compFuncGradDtm(curTimeSeries, pcDim, pdTwoList[scaleListCnt], homDim, strictDim, maxEdgeLen, kNN, tdeSkip = 1, tdeDelay = 1)

        curHaarDerMat = haarDerMatList[scaleListCnt]

        gradVec += np.dot(curHaarDerMat.toarray(), curGradVec) 

        funcVal += curFuncVal

        scaleListCnt += 1

    return funcVal, gradVec

def compFGRipsHaar(timeSeriesData, pcDim, scaleList, haarDerMatList, pdTwoList, homDim, strictDim, maxEdgeLen, tdeSkip = 1, tdeDelay = 1):

    tsSiz = len(timeSeriesData) #At an earlier stage, it has been made a power of 2.

    numScales = len(scaleList)

    funcVal = 0

    gradVec = np.zeros((tsSiz,1)) 

    scaleListCnt = 0

    for iScale in scaleList:
        levSiz = int(tsSiz/iScale) #size of time series at this level. For Haar wavelets, it is also the number of wavelets.  

        curTimeSeries = np.zeros(levSiz)

        for ind in range(levSiz):
            for iSamp in range(iScale):
                curTimeSeries[ind] += timeSeriesData[ind*iScale + iSamp]

            curTimeSeries[ind] /= iScale 

        curFuncVal, curGradVec = compFuncGradRips(curTimeSeries, pcDim, pdTwoList[scaleListCnt], homDim, strictDim, maxEdgeLen)

        curHaarDerMat = haarDerMatList[scaleListCnt]

        gradVec += np.dot(curHaarDerMat.toarray(), curGradVec) 

        funcVal += curFuncVal

        scaleListCnt += 1

    return funcVal, gradVec

"""
Return the 2-Wasserstein distance between two PDs.
PD specified as a list of tensors.
"""
def comp2Wass(pdOne, pdTwo, homDim, strictDim):
    #return function value
    #pdOne: first PD.
    #pdTwo: second PD.
    #Both PDs specified as a list of tensors.
    #homDim: Homology dimensions over which the cost is computed.
    #strictDim: Homology dimensions over which strict one-to-one match is enforced. 

    funcVal = 0

    for iHomDim in range(len(homDim)):
        pdOnePick = pdOne[iHomDim]
        pdTwoPick = pdTwo[iHomDim] 

        if homDim[iHomDim] in strictDim: #enforce strict one-to-one match. Note for that homology dimension, the two PDs should have same number of points. 
            pdAugOnePick = pdOnePick
            pdAugTwoPick = pdTwoPick
            matchPts = findMatchConstrainedTensor(pdOnePick, pdTwoPick) #function to match equal number of points. 
        else:
            pdAugOnePick, pdAugTwoPick, numPtsOne, numPtsTwo = augmentPdTensor(pdOnePick, pdTwoPick) #augment PDs with diagonal projections from other PD. 
            matchPts = findMatchTensor(pdAugOnePick, pdAugTwoPick, numPtsOne, numPtsTwo, "solve_dense")
            #numPtsOne and numPtsTwo are the number of points for given homology dimension in the original persistence diagrams.
            #It is returned by augmentPd and needed by findMatch.

        #numPtsOne and numPtsTwo are the number of points for given homology dimension
        #in the original persistence diagrams. Only numPtsOne is needed in the following lines.
        numPtsOne = int(0.5*len(pdOnePick))

        for iPtPD in range(numPtsOne):
            funcVal = funcVal + (pdOnePick[2*iPtPD] - pdAugTwoPick[2*matchPts[iPtPD]])**2 + (pdOnePick[2*iPtPD + 1] - pdAugTwoPick[2*matchPts[iPtPD] + 1])**2

        numPtsOneTotal = int(0.5*len(pdAugTwoPick))

        #taking care of points in pdTwo that are matched to their projection on the diagonal
        for iPtPd in range(numPtsOne,numPtsOneTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
            if (matchPts[iPtPd] < numPtsTwo):
                diagVal = 0.5*(pdAugTwoPick[2*matchPts[iPtPd]] + pdAugTwoPick[2*matchPts[iPtPd] + 1])
                funcVal = funcVal + (pdAugTwoPick[2*matchPts[iPtPd]] - diagVal)**2 + (pdAugTwoPick[2*matchPts[iPtPd] + 1] - diagVal)**2
 
    return funcVal

"""
Return the 2-Wasserstein distance between two PDs.
PD specified as a list of tensors.
"""
def comp2WassOneDim(pdOne, pdTwo):
    #return function value
    #pdOne: first PD.
    #pdTwo: second PD.
    #Both PDs specified as a list of tensors.
    #homDim: Homology dimensions over which the cost is computed.
    #strictDim: Homology dimensions over which strict one-to-one match is enforced. 

    funcVal = 0

    #pdOneNP = pdOne.detach().numpy().copy()
    #pdTwoNP = pdTwo.detach().numpy().copy()

    matchPts = findMatchConstrainedTensor(pdOne, pdTwo) #function to match equal number of points. 

    #pdAugOne, pdAugTwo, numPtsOne, numPtsTwo = augmentPdTensor(pdOne, pdTwo) #augment PDs with diagonal projections from other PD. 
    #matchPts = findMatchTensor(pdAugOnePick, pdAugTwoPick, numPtsOne, numPtsTwo, "solve_dense")

    #numPtsOne and numPtsTwo are the number of points for given homology dimension in the original persistence diagrams.
    #It is returned by augmentPd and needed by findMatch.

    #numPtsOne and numPtsTwo are the number of points for given homology dimension
    #in the original persistence diagrams. Only numPtsOne is needed in the following lines.
    numPtsOne = int(0.5*len(pdOne))

    for iPtPD in range(numPtsOne):
        funcVal = funcVal + (pdOne[2*iPtPD] - pdTwo[2*matchPts[iPtPD]])**2 + (pdOne[2*iPtPD + 1] - pdTwo[2*matchPts[iPtPD] + 1])**2

    numPtsOneTotal = int(0.5*len(pdTwo))

    #taking care of points in pdTwo that are matched to their projection on the diagonal
    for iPtPd in range(numPtsOne,numPtsOneTotal): #if PD was not augmented by diagonal projection points, then this loop will not be entered into.
        if (matchPts[iPtPd] < numPtsTwo):
            diagVal = 0.5*(pdTwo[2*matchPts[iPtPd]] + pdTwo[2*matchPts[iPtPd] + 1])
            funcVal += (pdTwo[2*matchPts[iPtPd]] - diagVal)**2 + (pdTwo[2*matchPts[iPtPd] + 1] - diagVal)**2
 
    return funcVal
