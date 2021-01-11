#!/usr/bin/env python

import h5py
import numpy as np
import sys
import normal2D as n2d
import matplotlib.pyplot as plt

def fit2D(hdfChainFileName, fitFileName, plot=False):
    f = h5py.File(hdfChainFileName, 'r')
    dsetPos = f['chain']['position']
    dsetNames = f['chain']['names']

    for i,name in enumerate(dsetNames):
        if name==b'teff':
            iTeff = i
        if name==b'logg':
            iLogg = i

    (scale, mu0, mu1, s0, s1, thetaCov) = n2d.fitNormal2D(dsetPos[:,iTeff], dsetPos[:,iLogg])

    fout = open(fitFileName, 'w')
    print('teff_0     logg_0           teff_cov   logg_cov  theta_cov', file=fout)
    print(mu0, mu1, s0, s1, thetaCov, file=fout)
    fout.close()

    if plot:
        plt.plot(dsetPos[:,iTeff], dsetPos[:,iLogg], '.', ms=1, alpha=0.01)
        fit2d = n2d.normal2D(scale, mu0, mu1, s0, s1, thetaCov)
        fit2d.plotN2d()
        plt.show()

if __name__ == '__main__':
    fit2D(sys.argv[1], sys.argv[2], True)
    
    

