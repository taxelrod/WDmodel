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

import json
import pickle
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

        self.CRNLbandName = paramDict['CRNL']['bandName']

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
        try:
            self.av_lb, self.av_ub = objList[objName]['av']['bounds'] # use bounds specific for this object if present
        except KeyError:
            self.av_lb, self.av_ub = paramDict['av']['bounds'] # else use global bounds

        self.lowerBounds = np.array((self.teff_lb, self.logg_lb, self.av_lb))
        self.upperBounds = np.array((self.teff_ub, self.logg_ub, self.av_ub))
            

    def loadPhotometry(self):
        self.phot = ioWD.get_phot_for_obj(self.objName, self.photFileName)
        # ignore bands we don't want
        try:
            ignore_pb = self.paramDict['ignore_band']
        except KeyError:
            ignore_pb = None

        if ignore_pb is not None:
            pbnames = self.phot.pb
            pbnames = list(set(pbnames) - set(ignore_pb))
            # filter the photometry recarray to use only the passbands we want
            useind = [x for x, pb in enumerate(self.phot.pb) if pb in pbnames]
            useind = np.array(useind)
            self.phot = self.phot.take(useind)

            # set the pbnames from the trimmed photometry recarray to preserve order
            pbnames = list(self.phot.pb)


        # initialize self.pb
        self.nBands = len(self.phot.pb)
        self.pb = passband.get_pbmodel(self.phot.pb,self.model,  None)

        # set iCRNL to flag band that CRNL is applied to
        pbNames = np.array(list(self.pb.keys()))  # pbNames is an odict, hence this ugliness
        self.iCRNL = np.where(pbNames==self.CRNLbandName)[0][0]  

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

    def logPrior(self, teff, logg, av):
        # any parameter out of bounds, return -np.inf
        if teff < self.teff_lb or teff > self.teff_ub:
            return -np.inf
        if logg < self.logg_lb or logg > self.logg_ub:
            return -np.inf
        if av < self.av_lb or av > self.av_ub:
            return -np.inf
        
        # evaluate the 2D normal
        return np.log(self.tloggPrior.pdf(teff, logg))

    def calcSynMags(self, teff, logg, Av, deltaZp):
        modelSed = self.model._get_model(teff, logg)
        modelSed = self.model.reddening(self.model._wave, modelSed, Av)
        sedPack = np.rec.array([self.model._wave, modelSed], dtype=[('wave', '<f8'), ('flux', '<f8')])
                        # needed due to GN interface inconsistency
        self.synMags = passband.get_model_synmags(sedPack, self.pb) # recarray with dtype=[('pb', '<U5'), ('mag', '<f8')])
        self.synMags['mag'] += deltaZp
        self.optDM = np.sum((self.photCRNL-self.synMags['mag'])/self.phot['mag_err']**2)/np.sum(1./self.phot['mag_err']**2)
        self.synMags['mag'] += self.optDM

    def logLikelihood(self, teff, logg, Av, deltaZp, CRNL):
        # modify phot for CRNL for CRNL band, usually F160W
        self.photCRNL = np.copy(self.phot['mag'])
        self.photCRNL[self.iCRNL] += CRNL[1]*(self.photCRNL[self.iCRNL] - CRNL[0])
        self.calcSynMags(teff, logg, Av, deltaZp)
        return np.sum(-((self.photCRNL-self.synMags['mag'])/self.phot['mag_err'])**2)

    def logPost(self, teff, logg, Av, deltaZp, CRNL):

        self.teff = teff
        self.logg = logg
        self.Av = Av

        prior =  self.logPrior(teff, logg, Av)

        if not np.isfinite(prior):
            return -np.inf, None
        
        logLikelihood = self.logLikelihood(teff, logg, Av, deltaZp, CRNL)
        blob = self.photCRNL - self.synMags['mag']

        return logLikelihood + prior, blob

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

        return np.array((guess_teff, guess_logg, guess_Av)), \
            np.array((guess_teff_sigma, guess_logg_sigma, guess_Av_sigma))
            

class objectCollectionPhotometry(object):

    def __init__(self, paramDict):

        self.paramDict = paramDict
        self.objNames = list(paramDict['objList'])  # keeps it pickleable
        self.nObj = len(self.objNames)
        self.nObjParams = 3  # increase this if additional per-object variables need to be added, eg Rv
        self.objPhot = {}
        self.objSlice = {}
        self.blobSlice = {}
 
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
            jLo = i*self.nBands
            jHi = jLo + self.nBands
            self.blobSlice[objName] = np.s_[jLo:jHi]


        self.ZpSlice = np.s_[iHi:iHi+self.nBands-2]  # last element of deltaZp is not explicitly carried because deltaZp sume to 0
        self.CRNLSlice = np.s_[iHi+self.nBands-2:iHi+self.nBands] # CRNL0, CRNL1
        self.nParams = self.nObj*self.nObjParams + self.nBands - 2 + 2 # yes, I know!
        self.lowerBounds = np.zeros((self.nParams))
        self.upperBounds = np.zeros((self.nParams))
        for (i, objName) in enumerate(self.objNames):
            self.lowerBounds[self.objSlice[objName]] = self.objPhot[objName].lowerBounds
            self.upperBounds[self.objSlice[objName]] = self.objPhot[objName].upperBounds

        self.lowerBounds[self.ZpSlice], self.upperBounds[self.ZpSlice]  = paramDict['deltaZp']['bounds']

        print('------------------------------', self.nParams, self.CRNLSlice, self.lowerBounds[self.CRNLSlice], np.array([paramDict['CRNL']['bounds0'][0],paramDict['CRNL']['bounds1'][0]]))
        
        self.lowerBounds[self.CRNLSlice]  = np.array([paramDict['CRNL']['bounds0'][0],paramDict['CRNL']['bounds1'][0]])
        self.upperBounds[self.CRNLSlice]  = np.array([paramDict['CRNL']['bounds0'][1],paramDict['CRNL']['bounds1'][1]])

    # return an initial guess at theta
    
    def firstGuess(self):
        self.guess = np.zeros((self.nParams))
        self.guess_sigma = np.zeros((self.nParams))
        for objName in self.objNames:
            self.guess[self.objSlice[objName]], self.guess_sigma[self.objSlice[objName]]  = self.objPhot[objName].firstGuess()

        self.guess[self.ZpSlice] = 0
        self.guess_sigma[self.ZpSlice] = 1.0  # need to set this with a parameter

        self.guess[self.CRNLSlice] = 0 # midpoint of bounds?
        self.guess_sigma[self.CRNLSlice] = 1.0  # need to set this with a parameter

    
    # calculate the prior associated with model-wide parameters, deltaZp and CRNL for now
    
    def outerLogPrior(self, deltaZp, CRNL):

        deltaZpMin = np.amin(deltaZp)
        deltaZpMax = np.amax(deltaZp)

        if deltaZpMin < self.paramDict['deltaZp']['bounds'][0] or deltaZpMax > self.paramDict['deltaZp']['bounds'][1]:
            return -np.inf

        if CRNL[0] < self.paramDict['CRNL']['bounds0'][0] or CRNL[1] < self.paramDict['CRNL']['bounds1'][0]:
            return -np.inf

        if CRNL[0] > self.paramDict['CRNL']['bounds0'][1] or CRNL[1] > self.paramDict['CRNL']['bounds1'][1]:
            return -np.inf

        return 0 # uniform within bounds
        
    # this is what's called by emcee EnsembleSampler to get the logPosterior

    def __call__(self, theta):
        # unpack theta into self.nObj arrays of 3, to be interpreted by each objPhot + an array of length self.nBands, which
        # becomes deltaZp + an array of length 2, which is CRNL

        self.theta = theta

        # expand deltaZp by two elements.  The last element, deltaZPF160W is forced to always be zero.
        # the second to last element enforces sum(deltaZp)=0
        
        deltaZp = np.resize(theta[self.ZpSlice], (self.nBands)) # extend by two elements
        deltaZp[-1] = 0 # F160W
        deltaZp[-2] = -np.sum(theta[self.ZpSlice])

        CRNL = theta[self.CRNLSlice]  # two element array
        
        logPost = 0
        blob = np.zeros((self.nObj*self.nBands))
        for objName in self.objNames:
            objSlice = self.objSlice[objName]
            objBlobSlice = self.blobSlice[objName]
            obj = self.objPhot[objName]
            (teff, logg, Av) = theta[objSlice]
            objPost, objBlob = obj.logPost(teff, logg, Av, deltaZp, CRNL)
            logPost += objPost
            blob[objBlobSlice] = objBlob

        outerLogPrior = self.outerLogPrior(deltaZp, CRNL)

        if not np.isfinite(outerLogPrior):
            return -np.inf, blob
        
 #       print('blob, shape: ',blob, blob.shape)
        return logPost + outerLogPrior, blob 
    
        
# setupPhotEnv sets the environment variable PYSYN_CDBS prior to importing bandpass

def setupPhotEnv(pbPath):
    global passband
    
    if pbPath is None:
#        pbPath = '/home/tsa/Dropbox/WD/PyWD/WDmodel/WDdata/photometry/synphot/'
        pbPath = '/home/tim/WDmodel/WDdata/photometry/synphot/'

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
    nbands = objCollection.nBands
    nobj = objCollection.nObj

    outFileName = objCollection.paramDict['output_file']
    summaryFileName = objCollection.paramDict['summary_file']
    outf = h5py.File(outFileName, 'w')

    pickleFileName = objCollection.paramDict['pickle_file']
    
    # initialize chains

    objCollection.firstGuess()
    pos = emcee.utils.sample_ball(objCollection.guess, objCollection.guess_sigma, size=nwalkers)
    pos = enforceBounds(pos, objCollection.lowerBounds, objCollection.upperBounds) 
    
    # burnin
    print('starting burnin, nParams=', objCollection.nParams, objCollection.guess.shape, objCollection.lowerBounds.shape, objCollection.upperBounds.shape, pos.shape)

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
    theta        = map_samples[max_ind]

    # reset the sampler
    sampler.reset()

    message = "\nMAP Parameters after Burn-in"
    print(message)
    print(theta)
    outputResult(objCollection, theta, None, None)

    # set up output for chains

    chain = outf.create_group("chain")


    # set walkers to start production at final burnin state
    pos = result.coords
    
    # write to disk before we start
    outf.flush()

    # since we're going to save the chain in HDF5, we don't need to save it in memory elsewhere
    # funny stuff to maintain backward compatibility with emcee2.x
    
    # production sample

    result = sampler.run_mcmc(pos, nprod, progress=True, store=True, skip_initial_state_check=True)
    samples        = sampler.get_chain(flat=True)
    samples_lnprob = sampler.get_log_prob(flat=True)
    blobs = sampler.get_blobs(flat=True)

    print('debug:',samples.shape, blobs.shape)
                
    
    print('production finished')
    
    dset_chain  = chain.create_dataset("position", data=samples)
    dset_lnprob = chain.create_dataset("lnprob", data=samples_lnprob)
    dset_blob = chain.create_dataset("magerr", data=blobs)

    outf.flush()
    outf.close()

    # find the MAP position after the production run

    map_samples        = samples.reshape(nwalkers, nprod, nparam)
    map_samples_lnprob = samples_lnprob.reshape(nwalkers, nprod)
    max_ind        = np.argmax(map_samples_lnprob)
    max_ind        = np.unravel_index(max_ind, (nwalkers, nprod))
    max_ind        = tuple(max_ind)
    theta        = map_samples[max_ind]

    message = "\nMAP Parameters after Production"
    print(message)
    print(theta)
    outputResult(objCollection, theta, summaryFileName, pickleFileName)

    return
   
def outputResult(objCollection, theta=None, outFileName=None, pickleFileName=None):

    if theta is not None:
        lnPost = objCollection(theta)

    if outFileName is not None:
        f = open(outFileName, 'w')
    else:
        f = sys.stdout

    firstIter = True
    for objName in objCollection.objNames:
        obj = objCollection.objPhot[objName]
        if firstIter:
            deltaStr = ''
            pbnames = obj.pb.keys()
            for pb in pbnames:
                deltaStr += 'delta' + pb + ' '
            print('#name teff logg Av DM ', deltaStr, file=f)
            firstIter = False
        print(objName, obj.teff,  obj.logg, obj.Av, obj.optDM, file=f, end=' ')
        for i in range(len(obj.phot)):
            print(obj.photCRNL[i]-obj.synMags['mag'][i], file=f, end=' ')
        print(file=f)

    if outFileName is not None:
        f.close()

    if pickleFileName is not None:
        fpkl = open(pickleFileName, 'wb')
        pickle.dump(objCollection, fpkl)
        fpkl.close()


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
    
    
            
