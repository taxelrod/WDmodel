#!/usr/bin/env python

import h5py
import numpy as np
import sys
import normal2D as n2d

def fit2D(hdfChainFileName, fitFileName):
    f = h5py.File(hdfChainFileName, 'r')
    dsetPos = f['chain']['position']
    dsetNames = f['chain']['names']

    for i,name in enumerate(dsetNames):
        if name==b'teff':
            iTeff = i
        if name==b'logg':
            iLogg = i

    (scale, mu0, mu1, cov00, cov01, cov11) = n2d.fitNormal2D(dsetPos[:,iTeff], dsetPos[:,iLogg])

    fout = open(fitFileName, 'w')
    print('teff_0     logg_0           teff_cov   teff_logg_cov   logg_cov', file=fout)
    print(mu0, mu1, cov00, cov01, cov11, file=fout)
    fout.close()

if __name__ == '__main__':
    fit2D(sys.argv[1], sys.argv[2])
    
    

