"""
Calculate the maximum absolute difference between all pairs of array elements
"""
import pandas as pd
import numpy as np

def maxPairDiff(arr):

    maxdiff = 0
    for (i, x) in enumerate(arr):
        for (j, y) in enumerate(arr):
            if i != j:
                maxdiff = max(abs(x-y), maxdiff)

    return maxdiff


def gnObjStats(gnfile, statColNames):

    gnDF=pd.read_table(gnfile,sep='\s+')
    (rows, cols) = gnDF.shape

    # find the max group ID - tricky because of NaNs
    badGrp = np.where(np.isnan(gnDF['GroupID']))
    badSet = set(badGrp[0])
    allSet = set(range(rows))

    goodGrp = list(allSet - badSet)
    maxGrp = int(max(gnDF['GroupID'][goodGrp]))

    print('#obj', end=' ')
    for col in statColNames:
        print('d'+col, end=' ')
    print()
    
    for i in range(maxGrp):
        grp = np.where(gnDF['GroupID']==i+1)
        grpDF = gnDF.iloc[grp[0]]
        objName = grpDF['obj'].iloc[0]
        print(objName,end=' ')
        for col in statColNames:
            print('%6.4f' % (maxPairDiff(grpDF[col])), end=' ')
        print()

def gnResidualStats(gnfile):
    gnObjStats(gnfile, ['rF275W', 'rF336W', 'rF625W', 'rF775W', 'rF160W'])

def gnResultStats(gnfile):
    gnObjStats(gnfile, ['teff', 'logg', 'av'])
