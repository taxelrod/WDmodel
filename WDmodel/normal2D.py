import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class normal2D(object):
    def __init__(self, scale, mu0, mu1, s0, s1, thetaCov):
        self.mu = np.array([mu0, mu1])
        self.u = np.array([[np.cos(thetaCov), np.sin(thetaCov)],[np.sin(thetaCov), -np.cos(thetaCov)]])
        self.s = np.diag([s0, s1])
        self.cov = np.matmul(self.u, np.matmul(self.s, self.u))
        self.invCov = np.linalg.inv(self.cov)
        self.detCov = np.linalg.det(self.cov)
        self.const = scale/(2*np.pi) * 1./np.sqrt(self.detCov)

    def pdf(self, x0, x1):
        x = np.array([x0, x1])
        dx = x - self.mu
        val = self.const * np.exp(-0.5*np.dot(dx, np.matmul(self.invCov, dx)))
        return val

    def plotN2d(self, sigMul=3, nxPts=20, nyPts=20):
        sigma = 1./np.sqrt(np.diag(self.invCov))
        xa = np.linspace(self.mu[0] - sigMul*sigma[0], self.mu[0] + sigMul*sigma[0], nxPts)
        ya = np.linspace(self.mu[1] - sigMul*sigma[1], self.mu[1] + sigMul*sigma[1], nyPts)
        (xGrid, yGrid) = np.meshgrid(xa, ya)
        xGridF = ndarray.flatten(xGrid)
        yGridF = ndarray.flatten(yGrid)
        zGridF = np.zeros_like(xGridF)
        for (i, x) in enumerate(xGridF):
            y = yGridF[i]
            zGridF[i] = self.pdf(x, y)

        zGrid = np.reshape(zGridF, (nxPts, nyPts))
        
        plt.contour(xGrid, yGrid, zGrid, colors=['black'])
                        
        


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
    
    (histo, xEdges, yEdges) = np.histogram2d(xPoints, yPoints, bins=[nxBins, nyBins], range=range, normed=True)
    xCen = (xEdges[:-1] + xEdges[1:])/2
    yCen = (yEdges[:-1] + yEdges[1:])/2
    return (histo, xCen, yCen)

def normal2DObjFunc(normalParams, dataHisto, xEdges, yEdges, debug=False):
    
    (scale, mu0, mu1, s0, s1, thetaCov) = normalParams
    n2d = normal2D(scale, mu0, mu1, s0, s1, thetaCov)
    obj = 0
    for i,x in enumerate(xEdges):
        for j,y in enumerate(yEdges):
            obj += (n2d.pdf(x, y) - dataHisto[i, j])**2
    if debug:
        print(obj, scale, mu0, mu1, s0, s1, thetaCov)
    return obj

#def fitnormal2D(samples, sampleParams, xname, yname):
# find xname and yname in list -> ix, iy

def fitNormal2D(xSamples, ySamples, nxBins=50, nyBins=50, debug=False):
    (sampleHisto, xCen, yCen) = binSamples(xSamples, ySamples, nxBins, nyBins)

    scale = 1
    mu0 = np.mean(xSamples)
    mu1 = np.mean(ySamples)
    
    cov = np.cov(np.vstack((xSamples, ySamples)))
    print(cov)
    # use svd to decompose cov into u x s x vh
    # u and vh are (identical) reflection matrices, parameterized by thetaCov, s is a diagonal scale matrix
    # see https://en.wikipedia.org/wiki/Rotations_and_reflections_in_two_dimensions
    
    u, s, vh = np.linalg.svd(cov)
    thetaCov = np.arcsin(u[0,1])
    s0 = s[0]
    s1 = s[1]
    

#    cov11 = np.var(ySamples)

    s0Min = 0.02*s0
    s1Min = 0.02*s1
    
    guessParams = (scale, mu0, mu1, s0, s1, thetaCov) 
    minResult = minimize(normal2DObjFunc, guessParams, method='L-BFGS-B', bounds=((0, None), (None, None), (None, None) , (s0Min, None), (s1Min, None), (-2*np.pi, 2*np.pi)), args=(sampleHisto, xCen, yCen), options={'gtol':1.0e-7})

    if debug:
        (scale, mu0, mu1, s0, s1, thetaCov) = minResult.x
        fit2D = normal2D(scale, mu0, mu1, s0, s1, thetaCov)
        return minResult, fit2D
    else:
        print(minResult)
        return minResult.x

def plotHisto2D(histo, xCen, yCen):
    plt.contour(xCen, yCen, histo)
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
#    return z
