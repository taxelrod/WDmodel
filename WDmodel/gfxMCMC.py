"""
Note:  before running need "from photMCMC import objectPhotometry, objectCollectionPhotometry"
"""
import h5py
import pickle
import corner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def makePlots(hdfFileName, objPickleName, nPlot=None, outFileName=None, mapOutFileName=None):

    if outFileName is None:
        outFileName = 'test.pdf'
    
    f=h5py.File(hdfFileName,'r')
    runData = f['chain']

    fpkl=open(objPickleName, 'rb')
    objRun = pickle.load(fpkl)
    fpkl.close()

    objNames = objRun.objNames

    bandNames = objRun.objPhot[objNames[0]].pb.keys()
    bands = {}
    for i, bandName in enumerate(bandNames):
        bands[bandName] = i

    nBands = len(bands)
    
    if objNames is None:
        objNames = runData['objnames']
        
    runPosition = runData['position']
    runLnProb = runData['lnprob']
    runMagerr = runData['magerr']
    (nPts, nObjnBands) = runMagerr.shape

    #
    # identify MAP point
    #
    mapIdx = np.argmax(runLnProb)
    mapTheta = runPosition[mapIdx, :] # should be nObj*nParams in length ??  + nBands - 1 + 2
    mapMagerr = runMagerr[mapIdx, :]
    print(mapIdx, runLnProb[mapIdx], mapTheta, mapMagerr)
    #
    # make slices
    #
    if nPlot is None:
        plotSlice = np.s_[:]
    else:
        plotSlice = np.s_[-nPlot:]

    objParams = {'Teff':0, 'logg':1, 'Av':2}
    nObjParams = len(objParams)

    assert(nObjnBands%nBands == 0)
    nObj = nObjnBands/nBands

    bandSlice = {}
    objSlice = {}
    objMagSlice = {}
    for band in bands.keys():
        indx = bands[band]
        bandSlice[band] = np.s_[indx::nBands]
    
    for i,obj in enumerate(objNames):
        objIdx = i*nObjParams
        bandIdx = i*nBands
        objSlice[obj] = np.s_[plotSlice,objIdx:objIdx+nObjParams]
        objMagSlice[obj] = np.s_[plotSlice, bandIdx:bandIdx+nBands]

    pdfOut = PdfPages(outFileName)

    if mapOutFileName is not None:
        fMap = open(mapOutFileName, 'w')
    else:
        fMap = None

    #
    # make object corner plots
    #
    mapPt = {}
    bandSigmas = {}
    bandMedians = {}
    paramSigmas = {}
    paramMedians = {}

    for obj in objNames:
        cornerData = np.hstack((runPosition[objSlice[obj]], runMagerr[objMagSlice[obj]]))
        mapPts = np.zeros((nObjParams+nBands))
        mapPts[0:nObjParams] = mapTheta[objSlice[obj][1]]
        mapPts[nObjParams:] = mapMagerr[objMagSlice[obj][1]]
        mapPt[obj] = mapPts
        bandSigmas[obj] = np.std(runMagerr[objMagSlice[obj]], axis=0)
        bandMedians[obj] = np.median(runMagerr[objMagSlice[obj]], axis=0)
        paramSigmas[obj] = np.std(runPosition[objSlice[obj]], axis=0)
        paramMedians[obj] = np.median(runPosition[objSlice[obj]], axis=0)
        fig=corner.corner(cornerData,labels=list(objParams)+list(bands), show_titles=True, truths=mapPts )
        fig.text(0.7,0.95,obj,fontsize=16)
        plt.savefig(pdfOut, format='pdf')
        plt.close(fig)

    
    #
    # make band delta zp  + CRNL corner plots
    #
    nBandsM2 = nBands - 2
    cornerData = runPosition[plotSlice, -nBandsM2-1:]
    #
    # kluge alert!
    #
    zpF775W = -np.sum( runPosition[plotSlice, -nBandsM2-1:-1], axis=1 )
    print('zpF775W:', runPosition[plotSlice, -nBandsM2-1:-1].shape, cornerData.shape, np.mean(zpF775W))

    (nRow, nCol) = cornerData.shape
    
    cornerDatax = np.hstack((cornerData, np.reshape(zpF775W, (nRow, 1))))
    
    zpLabels = []
    for (i,bandName) in enumerate(bandNames):
        if i == nBandsM2:
            break
        zpLabels.append('zp' + bandName)

    zpLabels.append('CRNL1')
    zpLabels.append('zpF775W')
    
    print('zp truths:', mapTheta[-nBandsM2:])
#    fig=corner.corner(cornerData,labels=zpLabels, show_titles=True, truths=mapTheta[-nBandsM1-2:] )
    fig=corner.corner(cornerDatax,labels=zpLabels, show_titles=True )
    plt.savefig(pdfOut, format='pdf')
    plt.close(fig)

    pdfOut.close()
    f.close()

    if fMap is not None:
        fileHdrLine = '# obj teff_map logg_map Av_map '
        for bandName in bandNames:
            fileHdrLine += 'r' + bandName + '_map '
        fileHdrLine += 'teff_med logg_med Av_med '
        for bandName in bandNames:
            fileHdrLine += 'r' + bandName + '_med '
        fileHdrLine += 'teff_sigma logg_sigma Av_sigma '
        for bandName in bandNames:
            fileHdrLine += 'r' + bandName + '_sigma '
        print(fileHdrLine, file=fMap)
        
        for obj in objNames:
            print(obj, end=' ', file=fMap)
            for n, v in enumerate(mapPt[obj]):
                print(v, end=' ', file=fMap)
            for n, v in enumerate(paramMedians[obj]):
                print(v, end=' ', file=fMap)
            for n, v in enumerate(bandMedians[obj]):
                print(v, end=' ', file=fMap)
            for n, v in enumerate(paramSigmas[obj]):
                print(v, end=' ', file=fMap)
            for n, v in enumerate(bandSigmas[obj]):
                print(v, end=' ', file=fMap)
            print(' ', file=fMap)
            
        fMap.close()

