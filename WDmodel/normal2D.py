import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class normal2D(object):
    def __init__(self, scale, mu0, mu1, cov00, cov01, cov11):
        self.mu = np.array([mu0, mu1])
        self.cov = np.array([[cov00, cov01], [cov01, cov11]])
        self.invCov = np.linalg.inv(self.cov)
        self.detCov = np.linalg.det(self.cov)
        self.const = scale/(2*np.pi) * 1./np.sqrt(self.detCov)

    def pdf(self, x0, x1):
        x = np.array([x0, x1])
        dx = x - self.mu
        val = self.const * np.exp(-0.5*np.dot(dx, np.matmul(self.invCov, dx)))
        return val


def binSamples(xPoints, yPoints, nxBins, nyBins, sigWidth=2.0):
    xMean = np.mean(xPoints)
    xStd = np.std(xPoints)
    xMin = xMean - sigWidth*xStd
    xMax = xMean + sigWidth*xStd

    yMean = np.mean(yPoints)
    yStd = np.std(yPoints)
    yMin = yMean - sigWidth*yStd
    yMax = yMean + sigWidth*yStd

    range=np.array([[xMin, xMax],[yMin,yMax]])
    
    return np.histogram2d(xPoints, yPoints, bins=[nxBins, nyBins], range=range, normed=True) # (histo, xEdges, yEdges)

def normal2DObjFunc(normalParams, dataHisto, xEdges, yEdges):
    
    (scale, mu0, mu1, cov00, cov01, cov11) = normalParams
    n2d = normal2D(scale, mu0, mu1, cov00, cov01, cov11)
    obj = 0
    for i,x in enumerate(xEdges):
        for j,y in enumerate(yEdges):
            obj += (n2d.pdf(x, y) - dataHisto[i, j])**2
    print(obj, scale, mu0, mu1, cov00, cov01, cov11)
    return obj

#def fitnormal2D(samples, sampleParams, xname, yname):
# find xname and yname in list -> ix, iy

def fitnormal2D(xSamples, ySamples, nxBins=50, nyBins=50):
    (sampleHisto, xEdges, yEdges) = binSamples(xSamples, ySamples, nxBins, nyBins)

    scale = 1
    mu0 = np.mean(xSamples)
    cov00 = np.var(xSamples)
    cov01 = 0
    mu1 = np.mean(ySamples)
    cov11 = np.var(ySamples)
    
    guessParams = (scale, mu0, mu1, cov00, cov01, cov11) 
    minResult = minimize(normal2DObjFunc, guessParams, method='L-BFGS-B', bounds=((0, None), (None, None), (None, None) , (0, None), (None, None), (0, None)), args=(sampleHisto, xEdges[:-1], yEdges[:-1]), options={'gtol':1.0e-7})

    (scale, mu0, mu1, cov00, cov01, cov11) = minResult.x
    fit2D = normal2D(scale, mu0, mu1, cov00, cov01, cov11)
    return minResult, fit2D

def plotHisto2D(histo, xEdges, yEdges):
    plt.contour(xEdges[:-1], yEdges[:-1], histo)
    plt.show()
                

def plotNormal2D(n2d, x, y):
    lenX = len(x)
    lenY = len(y)
    z = np.zeros((lenY, lenX))
    for i in range(lenX): 
        for j in range(lenY): 
            z[j, i]=n2d.pdf(x[i],y[j]) 
        
    plt.contour(x,y,z)
    plt.show()
    return z
