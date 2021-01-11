"""
Calculate the maximum absolute difference between all pairs of array elements
"""
import sys
import pandas as pd
import numpy as np

def maxPairDiff(arr):

    maxdiff = 0
    for (i, x) in enumerate(arr):
        for (j, y) in enumerate(arr):
            if i != j:
                maxdiff = max(abs(x-y), maxdiff)

    return maxdiff

"""
Analyze and print pairwise differences in list of quantities for a GN results.dat or
residuals.dat file after it has been self matched on obj by topcat
"""

def gnObjStats(gnfile, statColNames, outfile):

    if outfile is not None:
        file = open(outfile, 'w')
    else:
        file = sys.stdout

    gnDF=pd.read_table(gnfile,sep='\s+')
    (rows, cols) = gnDF.shape

    # find the max group ID - tricky because of NaNs
    badGrp = np.where(np.isnan(gnDF['GroupID']))
    badSet = set(badGrp[0])
    allSet = set(range(rows))

    goodGrp = list(allSet - badSet)
    maxGrp = int(max(gnDF['GroupID'][goodGrp]))

    print('#obj', end=' ', file=file)
    for col in statColNames:
        print('d'+col, end=' ', file=file)
    print(file=file)
    
    for i in range(maxGrp):
        grp = np.where(gnDF['GroupID']==i+1)
        grpDF = gnDF.iloc[grp[0]]
        objName = grpDF['obj'].iloc[0]
        print(objName,end=' ', file=file)
        for col in statColNames:
            print('%6.4f' % (maxPairDiff(grpDF[col])), end=' ', file=file)
        print(file=file)

def gnResidualStats(gnfile, outfile=None):
    gnObjStats(gnfile, ['rF275W', 'rF336W', 'rF625W', 'rF775W', 'rF160W'], outfile)

def gnResultStats(gnfile, outfile=None):
    gnObjStats(gnfile, ['teff', 'logg', 'av'], outfile)
