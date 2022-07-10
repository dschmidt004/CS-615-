import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import random
import csv
import PIL
from IPython.display import display
import cv2
import pickle as pickle
import os
import math





class Layer (ABC) :
    def __init__ (self): 
        self.__prevIn = [] 
        self.__prevOut = []
    def setPrevIn(self ,dataIn): 
        self.__prevIn = dataIn
    def setPrevOut( self , out ): 
        self.__prevOut = out
    def getPrevIn( self ): 
        return self.__prevIn 
    def getPrevOut( self ): 
        return self.__prevOut
    def backward (self, gradIn, eta=10**-4) :
        sg = self.gradient()
        return gradIn@self.gradient()    
    @abstractmethod
    def forward(self ,dataIn):
        pass
    @abstractmethod
    def gradient(self):
        pass
    
class InputLayer(Layer):
    def __init__(self,dataIn):
        super().__init__()
        self.mean = np.mean(dataIn, axis=0)
        self.std = np.std(dataIn, axis=0)
        self.std = np.where(self.std!=0,self.std,1)
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataIn = (dataIn-self.mean)/self.std
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        pass
    
class DropoutLayer(Layer):
    def __init__(self,dataIn=None):
        super().__init__()
        self.droppedNode = []
        self.probability = 0.4
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.droppedNode = np.random.rand(dataIn.shape[0], dataIn.shape[1]) > self.probability
        output = self.droppedNode * dataIn /self.probability
        return output        

    def gradient(self) :
        return self.droppedNode / self.probability

class FullyConnectedLayer ( Layer ) :
    
    def __init__(self,sizeIn,sizeOut):
        super().__init__()
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.bias = np.random.uniform(low=-10**-4, high=10**-4, size=(sizeOut))
        self.weights = np.random.uniform(low=-10**-4, high=10**-4, size=(sizeIn,sizeOut))

    def getWeights(self):
        return self.weights
        
    def setWeights (self,weights):
        self.weights = weights

    def getBias(self) :
        return self.bias

    def setBias(self,bias):
        self.bias = bias

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        prevOut = dataIn@self.getWeights()+self.getBias()
        self.setPrevOut(prevOut)
        return prevOut

    def gradient(self) :
        return self.getWeights().T

    def backward(self,gradIn,eta=0.0001):
        gradOut = super().backward(gradIn)
        x = gradIn.shape[0]
        djdw = (self.getPrevIn().T@gradIn)
        djdb = np.sum(gradIn,axis = 0)
        self.weights -= djdw*eta/x
        self.bias -= djdb*eta/x
        
        return gradOut
        
    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        gradOut = self.getWeights().T.copy()
        djdb = np.sum(gradIn,axis = 0)
        djdw = (self.getPrevIn().T@gradIn)
        epsilon = 1e-8
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        
        self.momentumDB = beta1*self.momentumDB + (1-beta1)*djdb
        self.velocityDB = beta2*self.velocityDB + (1-beta2)*(djdb*djdb)
        
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        
        biasNum = self.momentumDB/(1-beta1**t)
        biasDenom = self.velocityDB/(1-beta2**t)
        
        self.weights -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        self.bias -= eta*(biasNum/(np.sqrt(biasDenom)+epsilon))
        return gradOut

class SigmoidLayer(Layer ):
    def __init__ (self ,dataIn=None):
        super().__init__()
        
    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        dataIn = np.array(dataIn)
        prevOut = 1/(1+np.exp(-dataIn))
        super().setPrevOut(prevOut) 
        return prevOut

    def gradient(self): 
        pOut = self.getPrevOut()
        return ( pOut) * (1 - np.array(pOut))

class ReLuLayer(Layer ):
    def __init__ (self ,dataIn=None):
        super().__init__()

    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        prevOut = np.maximum(dataIn, 0)
        super().setPrevOut(prevOut)            
        return prevOut
    
    def gradient(self):
        pOut = self.getPrevOut()
        return 1. * (pOut > 0)
    
class SoftmaxLayer(Layer ):
    def __init__ (self ,dataIn=None):
        super().__init__()

    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        assert len(dataIn.shape) == 2
        s = np.max(dataIn, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(dataIn - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] 
        prevout =  e_x / div
        super().setPrevOut(prevout)
        return prevout
    
    def gradient(self):
        pOut = self.getPrevOut() 
        J = -pOut[..., None] * pOut[:, None, :]
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = pOut * (1. - pOut)
        J.sum(axis=1)
        return J
    
    def backward(self, gradIn, eta=10**-4):
        sg = self.gradient()
        return np.array(gradIn)@np.array(sg)

class TanhLayer(Layer ):
    def __init__ (self ,dataIn=None):
        super().__init__()

    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        dataIn = np.array(dataIn)
        prevOut = (np.exp(dataIn)-np.exp(-dataIn))/(np.exp(dataIn)+np.exp(-dataIn))
        super().setPrevOut(prevOut)
        return prevOut
        
    def gradient(self): 
        pOut = self.getPrevOut()
        return  (1 - (pOut ** 2))
    
class CrossEntropy( ):
    def __init__ (self ,dataIn=None):
        super().__init__()

    def eval(self, y, yhat):
        return -np.sum( y * np.log(yhat + np.finfo(float).eps))
    
    def gradient(self, y, yhat):
        return -(y / (yhat + np.finfo(float).eps))
    
class LeastSquares( ):
    def eval(self, y, yhat):
        return (np.transpose(y - yhat) @ (y - yhat))/y.shape[0] 
    
    def gradient(self, y, yhat):
        return -2 * (y-yhat)  

class LogLoss( ):
    def __init__ (self ,dataIn=None):
        super().__init__()

    def eval(self, y, yhat):
        return -1 * np.sum(y * np.log(yhat + np.finfo(float).eps) + (1 - y) * np.log(1-yhat + np.finfo(float).eps))
    
    def gradient(self, y, yhat):
        return  -1 * ((y-yhat) / ((yhat * (1 - yhat)) + np.finfo(float).eps))

    
def mape(actual, pred): 
    return np.mean(np.abs((actual-pred)/actual)) * 100 
        
def rsme(actual, pred): 
    return np.sqrt(np.mean((pred-actual)**2))

class ConvLayer2D(Layer):
    
    def __init__ (self ,dataIn=None):
        super().__init__()
        self.kernel = None
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0

    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        out = self.convo(dataIn, self.kernel)
        super().setPrevOut(out)
        return out

    def gradient(self, gradIn): 
        prevIn = super().getPrevIn()
        return self.convo(prevIn, gradIn)
    
    def convo(self, dataIn, kernel):
        out = np.zeros((((dataIn.shape[0]-kernel.shape[0]) + 1),((dataIn.shape[1]-kernel.shape[1])+1)) )
        i = 0
        j = 0
        while i+kernel.shape[0] <= dataIn.shape[0]:
            j = 0
            while j+kernel.shape[1] <= dataIn.shape[1]:
                subArry = dataIn[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                out[i][j] = (np.sum(subArry*kernel))
                j+=1
            i+=1
        return out

    def backward(self, gradIn, eta=0.0001): 
        prevIn = super().getPrevIn()
        temp = weights
        weights = self.convo(prevIn, gradIn)
        self.kernel -= eta*weights
        return temp
    
    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        temp = self.kernel.copy()
        prevIn = super().getPrevIn()
        djdw = self.convo(prevIn, gradIn)
        epsilon = 1e-8
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.kernel -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        return temp

class ConvLayerT(Layer):
    
    def __init__ (self ,dataIn=None):
        super().__init__()
        self.kernel = None
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0

    def forward(self ,dataIn, pad): 
        dataIn = np.pad(dataIn, pad, self.pad_with)
        
        super().setPrevIn(dataIn)
        out = self.convo(dataIn, self.kernel)
        super().setPrevOut(out)
        return out

    def gradient(self, gradIn): 
        prevIn = super().getPrevIn()
        return self.convo(prevIn, gradIn)

    def pad_with(self, vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
    
    def convo(self, dataIn, kernel):
        out = np.zeros((((dataIn.shape[0]-kernel.shape[0]) + 1),((dataIn.shape[1]-kernel.shape[1])+1)) )
        i = 0
        j = 0
        while i+kernel.shape[0] <= dataIn.shape[0]:
            j = 0
            while j+kernel.shape[1] <= dataIn.shape[1]:
                subArry = dataIn[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                out[i][j] = (np.sum(subArry*kernel))
                j+=1
            i+=1
        return out

    def backward(self, gradIn, eta=0.0001):
        prevIn = super().getPrevIn()
        weights = self.convo(prevIn, gradIn)
        self.kernel -= eta*weights
        return weights

    
    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        prevIn = super().getPrevIn()
        temp = self.kernel.copy()
        djdw = self.convo(prevIn, gradIn)
        epsilon = 1e-8
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.kernel -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        return self.convo(gradIn, temp)

class MaxPool(Layer):
    def __init__ (self ,dataIn=None):
        super().__init__()
        self.size = 2
        
    def forward(self, dataIn):
        super().setPrevIn(dataIn)
        ds_data = np.full((dataIn.shape[0] // self.size, dataIn.shape[1] // self.size), -float('inf'), dtype=dataIn.dtype)
        np.maximum.at(ds_data, (np.arange(dataIn.shape[0])[:, None] // self.size, np.arange(dataIn.shape[1]) // self.size), dataIn)
        self.setPrevOut(ds_data)
        return ds_data

    def gradient(self, gradIn): 
        prevIn = self.getPrevIn()
        grad = np.zeros(prevIn.shape)
        i = 0
        j = 0
        indxI = 0 
        while i < prevIn.shape[1]:
            j = 0
            while j < prevIn.shape[0]:
                indxI += 1
                try:
                    subArry = prevIn[i:i+self.size, j:j+self.size]
                    s1, s2 = np.unravel_index(subArry.argmax(), subArry.shape)
                    grad[s2+j][s1+i] = indxI
                except:
                    print(1234)
                j += self.size
            i += self.size
        return grad
    
    def backward(self, gradIn):
        prevIn = self.getPrevIn()
        grad = np.zeros(prevIn.shape)
        i = 0
        j = 0
        indxI = 0 
        while i < prevIn.shape[1]:
            j = 0
            while j < prevIn.shape[0]:
                try:
                    subArry = prevIn[j:j+self.size, i:i+self.size]
                    s1, s2 = np.unravel_index(subArry.argmax(), subArry.shape)
                    grad[s2+j][s1+i] = gradIn[indxI]
                except:
                    print(i)
                    print(j)
                    print(subArry)
                indxI += 1
                j += self.size
            i += self.size
        return grad        
    

def gaussianMatrix(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def space_to_depth(x, block_size):
    x = np.asarray(x)
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size,
                         reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z


def resizeImage(img):
    scale_percent = 50 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

path1 = "SampleMoviePosters/"
listing = os.listdir(path1)    
path2 = "ValidatePosters/"

validation = os.listdir(path2)    
im = None

inArry = []
for itm in listing:
    if itm == '.DS_Store':
        continue
    im = cv2.imread(path1 + itm,0) 
    im = im[:,:-1]
    im = resizeImage(im)
    
    inImg = resizeImage(im)
    
    inArry.append(inImg)

inArry = np.array(inArry)

inArry = np.reshape(inArry, (-1,45)) 

IPL = InputLayer(inArry)

RL = ReLuLayer()

L1a = ConvLayerT()

L1a.kernel =  gaussianMatrix(3)

L5 = LeastSquares()

lossArry = []
validArry = []
upLossArry = []

epoch = 50

for i in range(epoch):
    print(i)
    
    loss = 0
    count = 0

    for itm in listing:
        if itm == '.DS_Store':
            continue
        count +=1
        im = cv2.imread(path1 + itm,0) 
        im = im[:,:-1]
        im = resizeImage(im)

        inImg = resizeImage(im)

        y = np.ndarray.flatten(im)

        ## UPSAMPLE
        
        inImg = inImg.repeat(2, axis=0).repeat(2, axis=1)
        
        out1 = L1a.forward(inImg, 1)
        
        yhat = np.ndarray.flatten(out1)
        
        loss += np.sum(L5.eval(y, yhat))
        
        grad1 = np.array(L5.gradient(y, yhat))
        
        grad1 = np.reshape(grad1, (134,90))
        
        L1a.backwardADAM(i+1, grad1, 0.99,0.999,0.001) 

    lossArry.append(loss/count)
    
    upLoss = 0 
    loss = 0
    count = 0
    for itm in validation:
        if itm == '.DS_Store':
            continue
        count +=1
        im = cv2.imread(path2 + itm,0) 
        im = im[:,:-1]
        im = resizeImage(im)

        inImg = resizeImage(im)
        
        inImg = inImg.repeat(2, axis=0).repeat(2, axis=1)
        
        out1 = L1a.forward(inImg, 1)

        upScaleLoss = np.ndarray.flatten(inImg)
        upLoss += np.sum(L5.eval(y, upScaleLoss)) 
        
        yhat = np.ndarray.flatten(out1)
        loss += np.sum(L5.eval(y, yhat))        
        
        cv2.imwrite('start_' + str(itm) , inImg)
        cv2.imwrite('target_' + str(itm) , im)
        cv2.imwrite('end_' + str(i) + "_" + str(itm) , out1)   

    validArry.append(loss/count)
    upLossArry.append(upLoss/count)





plt.plot(range(len(lossArry)), lossArry, label="Train")
plt.plot(range(len(validArry)), validArry, label="Valid")   
plt.plot(range(len(upLossArry)), upLossArry, label="upscaleLoss")   
plt.title("")
plt.xlabel("epochs")
plt.ylabel("SME Loss Avg")
figure = plt.gcf()
figure.set_size_inches(6, 8)
plt.legend()
#plt.savefig("MNIST.png", dpi=200)
plt.show()  
