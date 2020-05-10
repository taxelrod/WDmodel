#!/usr/bin/env python
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

Must require same bands for all objects - check after instantiating all objects
"""

import json
import importlib
import ioWD
import os
import sys
import numpy as np
from astropy.table import Table
import normal2D
import WDmodel
import emcee
import h5py

# objectPhotometry encapsulates all photometric data and fit results for a WD

class objectPhotometry(object):
    def __init__(self, objName=None, paramDict=None):
        if objName is None:
            return
        
        self.objName = objName
        self.paramDict = paramDict
        objList = paramDict['objList']
        if objList is None:
            print('objectPhotometry init: objlist not in paramDict')
        elif objName not in objList:
            print('objectPhotometry init: ', objName, 'not in objlist')
        else:
            self.objParams = objList[objName]
            self.photFileName = self.objParams['photFile']
            self.tloggFileName = self.objParams['tloggFile']

        self.grid_file = paramDict['grid_file']
        self.grid_name = None
        self.model = WDmodel.WDmodel(self.grid_file, self.grid_name)

        self.teff_lb, self.teff_ub = paramDict['teff']['bounds']
        self.logg_lb, self.logg_ub = paramDict['logg']['bounds']
        self.av_lb, self.av_ub = paramDict['av']['bounds']
        self.dm_lb, self.dm_ub = paramDict['dm']['bounds']

        self.lowerBounds = np.array((self.teff_lb, self.logg_lb, self.av_lb, self.dm_lb))
        self.upperBounds = np.array((self.teff_ub, self.logg_ub, self.av_ub, self.dm_ub))
            

    def loadPhotometry(self):
        self.phot = ioWD.get_phot_for_obj(self.objName, self.photFileName)
        # initialize self.pb
        self.nBands = len(self.phot.pb)
        self.pb = passband.get_pbmodel(self.phot.pb,self.model,  None)

    def initPrior(self, tloggFileName=None):
        if tloggFileName is not None:
            self.tloggFileName = tloggFileName
        # read the tloggfile, setup the 2Dnormal based on its parameters
        self.tloggTable = Table.read(self.tloggFileName, format='ascii')
        colDict = self.tloggTable.columns
        self.teff_0 = colDict['teff_0'].data[0]
        self.logg_0 = colDict['logg_0'].data[0]
        self.teff_cov = colDict['teff_cov'].data[0]
        self.theta_cov = colDict['theta_cov'].data[0]
        self.logg_cov = colDict['logg_cov'].data[0]
        self.tloggPrior = normal2D.normal2D(1.0, self.teff_0, self.logg_0, self.teff_cov, self.logg_cov, self.theta_cov)

    def logPrior(self, teff, logg, av, dm):
        # any parameter out of bounds, return -np.inf
        if teff < self.teff_lb or teff > self.teff_ub:
            return -np.inf
        if logg < self.logg_lb or logg > self.logg_ub:
            return -np.inf
        if av < self.av_lb or av > self.av_ub:
            return -np.inf
        if dm < self.dm_lb or dm > self.dm_ub:
            return -np.inf
        
        # evaluate the 2D normal
        return -np.log(self.tloggPrior.pdf(teff, logg))

    def calcSynMags(self, teff, logg, Av, dm, deltaZp):
        modelSed = self.model._get_model(teff, logg)
        modelSed = self.model.reddening(self.model._wave, modelSed, Av)
        sedPack = np.rec.array([self.model._wave, modelSed], dtype=[('wave', '<f8'), ('flux', '<f8')])
                        # needed due to GN interface inconsistency
        self.synMags = passband.get_model_synmags(sedPack, self.pb) # recarray with dtype=[('pb', '<U5'), ('mag', '<f8')])
        self.synMags['mag'] += dm + deltaZp

    def logLikelihood(self, teff, logg, Av, dm, deltaZp):
        self.calcSynMags(teff, logg, Av, dm, deltaZp)
        return np.sum(-((self.phot['mag']-self.synMags['mag'])/self.phot['mag_err'])**2)

    def logPost(self, teff, logg, Av, dm, deltaZp):

        prior =  self.logPrior(teff, logg, Av, dm)

        if not np.isfinite(prior):
            return -np.inf
        
        return self.logLikelihood(teff, logg, Av, dm, deltaZp) + prior

    def firstGuess(self):

        try:
            guess_teff = self.objParams['guess_teff']   # if one is specified, both must be
            guess_teff_sigma = self.objParams['guess_teff_sigma']
        except KeyError:
            guess_teff = self.teff_0  # center of prior distribution
            guess_teff_sigma = np.sqrt(self.teff_cov)
            
        try:
            guess_logg = self.objParams['guess_logg']
            guess_logg_sigma = self.objParams['guess_logg_sigma']
        except KeyError:
            guess_logg = self.logg_0  # center of prior distribution
            guess_logg_sigma = np.sqrt(self.logg_cov)

        try:
            guess_Av = self.objParams['guess_Av']
            guess_Av_sigma = self.objParams['guess_Av_sigma']
        except KeyError:
            guess_Av = 0
            guess_Av_sigma = 0.5

        try:
            guess_dm = self.objParams['guess_dm']
            guess_dm_sigma = self.objParams['guess_dm_sigma']
        except KeyError:
            self.calcSynMags(guess_teff, guess_logg, guess_Av, 0, np.zeros((self.nBands)))
            guess_dm = np.mean(self.phot.mag - self.synMags.mag)
            guess_dm_sigma = np.std(self.phot.mag - self.synMags.mag)

        return np.array((guess_teff, guess_logg, guess_Av, guess_dm)), \
            np.array((guess_teff_sigma, guess_logg_sigma, guess_Av_sigma, guess_dm_sigma))
            

class objectCollectionPhotometry(object):

    def __init__(self, paramDict):

        self.paramDict = paramDict
        self.objNames = paramDict['objList'].keys()
        self.nObj = len(self.objNames)
        self.nObjParams = 4  # increase this if additional per-object variables need to be added, eg Rv
        self.objPhot = {}
        self.objSlice = {}
 
        print(self.objNames)
        for (i, objName) in enumerate(self.objNames):
            iLo = i*self.nObjParams
            iHi = iLo + self.nObjParams
            self.objSlice[objName] = np.s_[iLo:iHi]
            self.objPhot[objName] = objectPhotometry(objName, paramDict)
            self.objPhot[objName].loadPhotometry()
            self.objPhot[objName].initPrior()
            if i==0:
                self.nBands = len(self.objPhot[objName].pb)
            else:
                checkNbands = len(self.objPhot[objName].pb)
                assert checkNbands == self.nBands

        self.ZpSlice = np.s_[iHi:iHi+self.nBands-1]  # last element of deltaZp is not explicitly carried because deltaZp sume to 0
        self.nParams = self.nObj*self.nObjParams + self.nBands - 1
        self.lowerBounds = np.zeros((self.nParams))
        self.upperBounds = np.zeros((self.nParams))
        for (i, objName) in enumerate(self.objNames):
            self.lowerBounds[self.objSlice[objName]] = self.objPhot[objName].lowerBounds
            self.upperBounds[self.objSlice[objName]] = self.objPhot[objName].upperBounds

        self.lowerBounds[self.ZpSlice], self.upperBounds[self.ZpSlice]  = paramDict['deltaZp']['bounds']

    # return an initial guess at theta
    
    def firstGuess(self):
        self.guess = np.zeros((self.nParams))
        self.guess_sigma = np.zeros((self.nParams))
        for objName in self.objNames:
            self.guess[self.objSlice[objName]], self.guess_sigma[self.objSlice[objName]]  = self.objPhot[objName].firstGuess()

        self.guess[self.ZpSlice] = 0
        self.guess_sigma[self.ZpSlice] = 1.0  # need to set this with a parameter

    
    # this is what's called by emcee EnsembleSampler to get the logPosterior

    def __call__(self, theta):
        # unpack theta into self.nObj arrays of 4, to be interpreted by each objPhot + an array of length self.nBands, which
        # becomes deltaZp

        deltaZp = np.resize(theta[self.ZpSlice], (self.nBands)) # extend by one element, set last element to enforce sum(deltaZp) = 0
        deltaZp[-1] = -np.sum(theta[self.ZpSlice])
        logPost = 0
        for objName in self.objNames:
            (teff, logg, Av, dm) = theta[self.objSlice[objName]]
            logPost += self.objPhot[objName].logPost(teff, logg, Av, dm, deltaZp)

        return logPost # + prior for deltaZp
    
        
# setupPhotEnv sets the environment variable PYSYN_CDBS prior to importing bandpass

def setupPhotEnv(pbPath):
    global passband
    
    if pbPath is None:
        pbPath = '/home/tsa/Dropbox/WD/PyWD/WDmodel/WDdata/photometry/synphot/'

    os.environ['PYSYN_CDBS'] = pbPath

    passband = importlib.import_module('passband')

def enforceBounds(pos, lb, ub):

    (nVec, lenVec) = pos.shape
    for n in range(nVec):
        pos[n,:] = np.maximum(pos[n,:], lb)
        pos[n,:] = np.minimum(pos[n,:], ub)

    return pos

def doMCMC(objCollection):
    # get MCMC params out of paramDict
    nwalkers = objCollection.paramDict['nwalkers']
    nburnin = objCollection.paramDict['nburnin']
    nprod = objCollection.paramDict['nprod']

    outFileName = objCollection.paramDict['output_file']
    outf = h5py.File(outFileName, 'w')
    
    # initialize chains

    objCollection.firstGuess()
    pos = emcee.utils.sample_ball(objCollection.guess, objCollection.guess_sigma, size=nwalkers)
    pos = enforceBounds(pos, objCollection.lowerBounds, objCollection.upperBounds) 
    
    # burnin
    print('starting burnin')
    sampler = emcee.EnsembleSampler(nwalkers, objCollection.nParams, objCollection)  # note that objCollection() returns the posterior
    result = sampler.run_mcmc(pos, nburnin, progress=True)
    print('burnin finished')

    # find the MAP position after the burnin

    nparam = objCollection.nParams
    samples        = sampler.get_chain(flat=True)
    samples_lnprob = sampler.get_log_prob(flat=True)
    map_samples        = samples.reshape(nwalkers, nburnin, nparam)
    map_samples_lnprob = samples_lnprob.reshape(nwalkers, nburnin)
    max_ind        = np.argmax(map_samples_lnprob)
    max_ind        = np.unravel_index(max_ind, (nwalkers, nburnin))
    max_ind        = tuple(max_ind)
    p1        = map_samples[max_ind]

    # reset the sampler
    sampler.reset()

    message = "\nMAP Parameters after Burn-in"
    print(message)
    print(p1)

    # set up output for chains

    chain = outf.create_group("chain")
    '''
    # save the parameter names corresponding to the chain
    free_param_names = np.array([str(x) for x in free_param_names])
    dt = free_param_names.dtype.str.lstrip('|').replace('U','S')
    chain.create_dataset("names",data=free_param_names.astype(np.string_), dtype=dt)
    '''


    # set walkers to start production at final burnin state
    pos = result.coords
    
    # write to disk before we start
    outf.flush()

    # since we're going to save the chain in HDF5, we don't need to save it in memory elsewhere
    # funny stuff to maintain backward compatibility with emcee2.x
    
    # production sample

    result = sampler.run_mcmc(pos, nprod, progress=True, store=True)
    samples        = sampler.get_chain(flat=True)
    samples_lnprob = sampler.get_log_prob(flat=True)
    
    print('production finished')
    
    dset_chain  = chain.create_dataset("position",(nwalkers*nprod,nparam),maxshape=(None,nparam), data=samples)
    dset_lnprob = chain.create_dataset("lnprob",(nwalkers*nprod,),maxshape=(None,), data=samples_lnprob)

    outf.flush()
    outf.close()

    # find the MAP position after the production run

    map_samples        = samples.reshape(nwalkers, nprod, nparam)
    map_samples_lnprob = samples_lnprob.reshape(nwalkers, nprod)
    max_ind        = np.argmax(map_samples_lnprob)
    max_ind        = np.unravel_index(max_ind, (nwalkers, nprod))
    max_ind        = tuple(max_ind)
    p1        = map_samples[max_ind]

    message = "\nMAP Parameters after Production"
    print(message)
    print(p1)

    return
   


def main(paramFileName, pbPath = None):

    setupPhotEnv(pbPath)

    f = open(paramFileName, 'r')
    paramDict = json.load(f)
    f.close()

    objCollection = objectCollectionPhotometry(paramDict)

    doMCMC(objCollection)

    return objCollection

if __name__ == '__main__':
    main(sys.argv[1])
    
    
            
