import h5py
import corner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def makePlots(hdfFileName, objNames=None, nPlot=None, outFileName=None):

    if outFileName is None:
        outFileName = 'test.pdf'
    
    f=h5py.File(hdfFileName,'r')
    runData = f['chain']
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
    mapTheta = runPosition[mapIdx, :] # should be nObj*nParams in length
    mapMagerr = runMagerr[mapIdx, :]
    print(mapIdx, runLnProb[mapIdx], mapTheta, mapMagerr)
    #
    # make slices
    #
    if nPlot is None:
        plotSlice = np.s_[:]
    else:
        plotSlice = np.s_[-nPlot:]
        
    bands = {'F275W':0, 'F336W':1, 'F475W':2, 'F625W':3, 'F775W':4, 'F160W':5}
    nBands = len(bands)

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

    #
    # make object corner plots
    #
    for obj in objNames:
        cornerData = np.hstack((runPosition[objSlice[obj]], runMagerr[objMagSlice[obj]]))
        mapPts = np.zeros((nObjParams+nBands))
        mapPts[0:nObjParams] = mapTheta[objSlice[obj][1]]
        mapPts[nObjParams:] = mapMagerr[objMagSlice[obj][1]]
        print(obj, mapPts)
        fig=corner.corner(cornerData,labels=list(objParams)+list(bands), show_titles=True, truths=mapPts )
        fig.text(0.7,0.95,obj,fontsize=16)
        plt.savefig(pdfOut, format='pdf')
        plt.close(fig)

    #
    # make band delta zp corner plots
    #
    nBandsM1 = nBands - 1
    cornerData = runPosition[plotSlice, -nBandsM1:]
    zpLabels = ['zp275W', 'zp336W', 'zp475W', 'zp625W', 'zp775W']
    print('zp truths:', mapTheta[-nBandsM1:])
    fig=corner.corner(cornerData,labels=zpLabels, show_titles=True, truths=mapTheta[-nBandsM1:] )
    plt.savefig(pdfOut, format='pdf')
    plt.close(fig)

    pdfOut.close()
    f.close()

