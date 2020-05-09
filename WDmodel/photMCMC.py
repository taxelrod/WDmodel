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
    
    # initialize chains
    '''
    # get the starting position and the scales for each parameter
    init_p0  = lnlike.get_parameter_dict()
    p0       = list(init_p0.values())
    free_param_names = list(init_p0.keys())
    std = [params[x]['scale'] for x in free_param_names]

    # create a sample ball
    pos = emcee.utils.sample_ball(p0, std, size=ntemps*nwalkers)
    pos = fix_pos(pos, free_param_names, params)

    if samptype == 'ensemble':
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnpost,\
                a=ascale,  pool=pool)
        ntemps = 1

    '''
    objCollection.firstGuess()
    pos = emcee.utils.sample_ball(objCollection.guess, objCollection.guess_sigma, size=nwalkers)
    pos = enforceBounds(pos, objCollection.lowerBounds, objCollection.upperBounds) 
    
    # burnin
    print('starting burnin')
    sampler = emcee.EnsembleSampler(nwalkers, objCollection.nParams, objCollection)  # note that objCollection() returns the posterior
    sampler.run_mcmc(pos, nburnin, progress=True)
    print('burnin finished')

    # find the MAP position after the burnin

    nparam = objCollection.nParams
    samples        = sampler.flatchain
    samples_lnprob = sampler.lnprobability
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
    '''
        # set walkers to start production at final burnin state
        try:
            pos = result[0]
        except TypeError:
    '''
    # production sample
    '''
    with progress.Bar(label="Production", expected_size=laststep+nprod, hide=False) as bar:
        bar.show(laststep)
        j = laststep
        for i, result in enumerate(sampler.sample(pos, iterations=thin*nprod, **sampler_kwargs)):
            if (i+1)%thin != 0:
                continue
            try:
                position = result[0]
                lnpost   = result[1]
            except:
                position = result.coords
                lnpost   = result.log_prob
                
            position = position.reshape((-1, nparam))
            lnpost   = lnpost.reshape(ntemps*nwalkers)
            dset_chain[ntemps*nwalkers*j:ntemps*nwalkers*(j+1),:] = position
            dset_lnprob[ntemps*nwalkers*j:ntemps*nwalkers*(j+1)] = lnpost

            # save state every 100 steps
            if (j+1)%100 == 0:
                # make sure we know how many steps we've taken so that we can resize arrays appropriately
                chain.attrs["laststep"] = j+1
                outf.flush()

                # save the state of the chain
                with open(statefile, 'wb') as f:
                    pickle.dump(result, f, 2)

            bar.show(j+1)
            j+=1

        # save the final state of the chain and nprod, laststep
        chain.attrs["nprod"]    = laststep+nprod
        chain.attrs["laststep"] = laststep+nprod
        with open(statefile, 'wb') as f:
            pickle.dump(result, f, 2)

    # save the acceptance fraction
    if resume:
        if "afrac" in list(chain.keys()):
            del chain["afrac"]
        if samptype != 'ensemble':
            if "tswap_afrac" in list(chain.keys()):
                del chain["tswap_afrac"]
    chain.create_dataset("afrac", data=sampler.acceptance_fraction)
    if samptype != 'ensemble' and ntemps > 1:
        chain.create_dataset("tswap_afrac", data=sampler.tswap_acceptance_fraction)

    samples         = np.array(dset_chain)
    samples_lnprob  = np.array(dset_lnprob)

    # finalize the chain file, close it and close the pool
    outf.flush()
    outf.close()
    '''
    # output
    '''
    # find the MAP value after production
    map_samples = samples.reshape(ntemps, nwalkers, laststep+nprod, nparam)
    map_samples_lnprob = samples_lnprob.reshape(ntemps, nwalkers, laststep+nprod)
    max_ind = np.argmax(map_samples_lnprob)
    max_ind = np.unravel_index(max_ind, (ntemps, nwalkers, laststep+nprod))
    max_ind = tuple(max_ind)
    p_final = map_samples[max_ind]
    lnlike.set_parameter_vector(p_final)
    message = "\nMAP Parameters after Production"
    print(message)

    for k, v in lnlike.get_parameter_dict().items():
        message = "{} = {:f}".format(k,v)
        print(message)
    message = "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    print(message)

    # return the parameter names of the chain, the positions, posterior, and the shape of the chain
    return  free_param_names, samples, samples_lnprob, everyn, (ntemps, nwalkers, laststep+nprod, nparam)
    '''

    return pos

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
    
    
            
