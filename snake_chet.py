import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import*
import skimage.filters as filt


###1\read the original image
Image = cv2.imread('2.bmp',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img = np.array(image,dtype = np.float)
plt.imshow(img,cmap = 'gray')
print(shape(img))

###2\define the initial snake
t = np.linspace(0,2*np.pi,60,endpoint = True)
y = 100+30*np.cos(t)
x = 100+30*np.sin(t)

####3\define the key parameters
alpha = 0.001
beta  = 0.4
gamma = 100
sigma = 20
iterations = 500

####\4define the matrix
N = np.size(x)
a = gamma*(2*alpha+6*beta)+1
b = gamma*(-alpha-4*beta)
c = gamma*beta

p = np.zeros((N,N),dtype=np.float)
p[0] = np.c_[a,b,c,np.zeros((1,N-5)),c,b]
print(p[0].shape)
for i in range(N):
    p[i] = np.roll(p[0],i)
p = np.linalg.inv(p)
#print("the matrix is:",p)

###5\define the external energy && computation
#img = cv2.GaussianBlur(img,(9,9),2)
#img = cv2.GaussianBlur(img,(9,9),2)

#Iy,Ix = np.gradient(img)
#gmi = (Iy**2+Ix**2)**0.5
#Iy,Ix = np.gradient(gmi)

#smoothed = filt.gaussian ((img - img.min())/(img.max()-img.min()),sigma)
smoothed = cv2.GaussianBlur((img-img.min()) / (img.max()-img.min()),(89,89),sigma)
giy,gix  = np.gradient(smoothed)
gmi = (gix**2+giy**2)**0.5
gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
Iy,Ix = np.gradient(gmi)

# def fx(x, y):
    # x[ x < 0 ] = 0.
    # y[ y < 0 ] = 0.

    # x[ x > img.shape[1]-1 ] = img.shape[1]-1
    # y[ y > img.shape[0]-1 ] = img.shape[0]-1

    # return ggmix[ (y.round().astype(int), x.round().astype(int)) ]

# def fy(x, y):
    # x[ x < 0 ] = 0.
    # y[ y < 0 ] = 0.

    # x[ x > img.shape[1]-1 ] = img.shape[1]-1
    # y[ y > img.shape[0]-1 ] = img.shape[0]-1

    # return ggmiy[ (y.round().astype(int), x.round().astype(int)) ]

def fmax(x,y):
    x[x < 0] = 0
    y[y < 0] = 0
    x[x > img.shape[1]-1] = img.shape[1]-1
    y[y > img.shape[0]-1] = img.shape[0]-1
    return y.round().astype(int),x.round().astype(int)

###6\computation & iteration
plt.plot(y,x,'.')
for i in range(iterations):
    fex = Ix[fmax(x,y)]
    fey = Iy[fmax(x,y)]
    x = np.dot(p,x+gamma*fex)
    y = np.dot(p,y+gamma*fey)
    if i%50 ==0:
        plt.plot(x,y,'.')
plt.show()


# for i in range(N):
    # if i < N-2:
        # p[i][i] = a
        # p[i][i+1] = b
        # p[i][i+2] = c
        # p[i][i-2] = c
        # p[i][i-1] = b
    # elif i == N-2:
        # p[i][i] =a
        # p[i][i+1] = b
        # p[i][i+2-N] = c
        # p[i][i-2] = c
        # p[i][i-1] = b
    # else :
        # p[i][i] =a
        # p[i][i+1-N] = b
        # p[i][i+2-N] = c
        # p[i][i-2] = c
        # p[i][i-1] = b

# plt.plot(y,x,'.')
# for i in range(iterations):
    # fex = 2*Ix[fmax(x,y)]
    # fey = 2*Iy[fmax(x,y)]
    # fex = 2*Ix[x.astype(int),y.astype(int)]
    # fey = 2*Iy[x.astype(int),y.astype(int)]
    # x = np.dot(p,x+gamma*fex)
    # y = np.dot(p,y+gamma*fey)
    # x,y = fmax(x,y)
    # if i%10 ==0:
        # plt.plot(y,x,'.')
# plt.show()


