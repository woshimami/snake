import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.filters as filt

def snake(img,x,y,alpha = 0.001,beta = 0.4,gamma = 100,
            sigma = 20, iterations =500):
    """
    The snake algorithm to segment image
    
    Parameters
    ------
    img : ndarray
        input image
    
    ------
    x,y : ndarry
        X-coordinate and Y-coordinate of the initial contour
    alpha,beta: number
        The set of parameters of internal energy
    gamma : number
        Parameter cotrolling the external engery
    sigma : number
        Standard deviation
    iterations : number
        The number of iteration
    """
    # compute the matrix
    N = np.size(x)
    a = gamma*(2*alpha+6*beta)+1
    b = gamma*(-alpha-4*beta)
    c = gamma*beta
    p = np.zeros((N,N),dtype = np.float)
    p[0] = np.c_[a,b,c,np.zeros((1,N-5)),c,b]
    for i in range(N):
        p[i] = np.roll(p[0],i)
    p = np.linalg.inv(p)
    # filter the image
    smoothed = cv2.GaussianBlur((img-img.min()) / (img.max()-img.min()),(89,89),sigma)
    giy,gix  = np.gradient(smoothed)
    gmi = (gix**2+giy**2)**0.5
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
    Iy,Ix = np.gradient(gmi)
    
    # avoid the curvature evolve to the outside of the image
    def fmax(x,y):
        x[x < 0] = 0
        y[y < 0] = 0
        x[x > img.shape[1]-1] = img.shape[1]-1
        y[y > img.shape[0]-1] = img.shape[0]-1
        return y.round().astype(int),x.round().astype(int)
    for i in range(iterations):
        fex = Ix[fmax(x,y)]
        fey = Iy[fmax(x,y)]
        x = np.dot(p,x + gamma*fex)
        y = np.dot(p,y + gamma*fey)
    return x,y

