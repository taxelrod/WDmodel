import h5py
import corner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def makePlots(hdfFileName, objNames=None, nPlot=None, outFileName=None):

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
    for band in bands.keys():
        indx = bands[band]
        bandSlice[band] = np.s_[indx::nBands]
    
    for i,obj in enumerate(objNames):
        objIdx = i*nObjParams
        objSlice[obj] = np.s_[plotSlice,objIdx:objIdx+nObjParams]

    pdfOut = PdfPages(outFileName)

    #
    # make object corner plots
    #
    for obj in objNames:
        fig=corner.corner(runPosition[objSlice[obj]],labels=list(objParams), show_titles=True)
        fig.text(0.7,0.95,obj,fontsize=16)
        plt.savefig(pdfOut, format='pdf')
        plt.close(fig)

    #
    # make band plots
    #
    for band in bands.keys():
        magerr = runMagerr[plotSlice, bandSlice[band]].flatten()
        fig = plt.figure()
        plt.hist(magerr, bins=20)
        fig.text(0.7, 0.95, band, fontsize=16)
        plt.savefig(pdfOut, format='pdf')
        plt.close(fig)
        
    pdfOut.close()
    f.close()
