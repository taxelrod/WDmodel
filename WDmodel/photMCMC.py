"""
Fit photometry of all objects simultaneously, using Teff-logg priors from previous
per-object spectrum analysis

Plan:

0. Have all file names in a .json param file
1. Input all photometry data files
x. Input all bandpass files (can we use GN code here?)
2. Input all teff-logg prior parameters
3. Construct the ln_liklihood and ln_posterior functions
4. Run MCMC

How to handle missing bands - for some objects only?

"""

"""
1 ------------
for objname in objList:
   phot = io.get_phot_for_obj(objname, photfile)   # main.py

x -------------
    pbnames = []
    if phot is not None:
        pbnames = np.unique(phot.pb)
        if excludepb is not None:
            pbnames = list(set(pbnames) - set(excludepb))

        # filter the photometry recarray to use only the passbands we want
        useind = [x for x, pb in enumerate(phot.pb) if pb in pbnames]
        useind = np.array(useind)
        phot = phot.take(useind)

        # set the pbnames from the trimmed photometry recarray to preserve order
        pbnames = list(phot.pb)

    pbs = passband.get_pbmodel(pbnames, model, pbfile=pbfile)  # main.py

3 ------------
    mod_spec = model._get_obs_model(self.teff, self.logg, self.av_spec, self.fwhm,\   # likelihood.py WDmodel_Likelihood.get_value()
            spec.wave, rv=self.rv, pixel_scale=pixel_scale)
    mod_phot, full = model._get_full_obs_model(self.teff, self.logg, self.av, self.fwhm,\
            spec.wave, rv=self.rv, pixel_scale=pixel_scale)
    mod_mags = get_model_synmags(full, pbs, mu=self.mu)
    phot_res = phot.mag - mod_mags.mag
    phot_chi = np.sum(phot_res**2./((phot.mag_err**2.)+(phot_dispersion**2.)))

====

Overall likelihood(theta) needs to unpack components into groups for each object (is this a use for rec.array?) + those common to all (deltaZp)
"""

import json
import importlib
import ioWD
import os
import numpy as np
from astropy.table import Table
import normal2D
import WDmodel


# objectPhotometry encapsulates all photometric data and fit results for a WD

class objectPhotometry(object):
    def __init__(self, objName, paramDict):
        self.objName = objName
        objList = paramDict['objList']
        if objList is None:
            print('objectPhotometry init: objlist not in paramDict')
        elif objName not in objList:
            print('objectPhotometry init: ', objName, 'not in objlist')
        else:
            self.photFileName = objList[objName]['photFile']
            self.tloggFileName = objList[objName]['tloggFile']

        self.grid_file = paramDict['grid_file']
        self.grid_name = None
        self.model = WDmodel.WDmodel(self.grid_file, self.grid_name)
            

    def loadPhotometry(self):
        self.phot = ioWD.get_phot_for_obj(self.objName, self.photFileName)
        # initialize self.pb
        self.nBands = len(self.phot.pb)
        self.pb = passband.get_pbmodel(self.phot.pb,self.model,  None)
        # create empty synmags array
        self.synMags = np.zeros((self.nBands))

    def initPrior(self):
        # read the tloggfile, setup the 2Dnormal based on its parameters
        self.tloggTable = Table.read(self.tloggFileName, format='ascii')
        colDict = self.tloggTable.columns
        self.teff_0 = colDict['teff_0'].data[0]
        self.logg_0 = colDict['logg_0'].data[0]
        self.teff_cov = colDict['teff_cov'].data[0]
        self.teff_logg_cov = colDict['teff_logg_cov'].data[0]
        self.logg_cov = colDict['logg_cov'].data[0]
        self.tloggPrior = normal2D.normal2D(1.0, self.teff_0, self.logg_0, self.teff_cov, self.teff_logg_cov, self.logg_cov)

    def lnPrior(self, teff, logg, Av, dm):
        # evaluate the 2D normal
        pass

    def calcSynMags(self, teff, logg, Av, dm, deltaZp):
        pass


class objectCollectionPhotometry(object):
    def __init__(self, paramDict):
        self.objectPhotList = []
        # bring init stuff from main in here

    # logPosterior is what's called by emcee EnsembleSampler
    def logPosterior(theta):
        return 0
    
# setupPhotEnv sets the environment variable PYSYN_CDBS prior to importing bandpass

def setupPhotEnv(pbPath):
    global passband
    
    if pbPath is None:
        pbPath = '/home/tsa/Dropbox/WD/PyWD/WDmodel/WDdata/photometry/synphot/'
    os.environ['PYSYN_CDBS'] = pbPath

    passband = importlib.import_module('passband')


def logLikelihood(theta, objPhot):
    pass

def main(paramFileName, pbPath = None):

    setupPhotEnv(pbPath)

    f = open(paramFileName, 'r')
    paramDict = json.load(f)
    f.close()

    objNames = paramDict['objList'].keys()

    objPhot = {}

    for objName in objNames:
        objPhot[objName] = objectPhotometry(objName, paramDict)
        objPhot[objName].loadPhotometry()
        objPhot[objName].initPrior()

    # need to initialize passbands here

    return objPhot
    
    
            
