import numpy as np
import cv2
import matplotlib.pyplot as plt
from snake import *

#read the image
Image = cv2.imread('2.bmp',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img = np.array(image,dtype = np.float)

#define the initial snake
t = np.linspace(0,2*np.pi,60,endpoint = True)
x_0 = 100+30*np.sin(t)
y_0 = 100+30*np.cos(t)

#plot the image and results
plt.imshow(img,cmap = 'gray')
x_1,y_1= snake(img,x_0,y_0)
plt.plot(x_0,y_0,'.')
plt.plot(x_1,y_1,'.')
plt.show()