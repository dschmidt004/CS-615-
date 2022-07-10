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
        return gradIn*self.gradient()    
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
    
    def reverse(self, dataIn):
        dataIn = (dataIn*self.std) + self.mean
        return dataIn
        
class DropoutLayer(Layer):
    def __init__(self,dataIn=None):
        super().__init__()
        self.layer = []
        self.droppedNode = []
        self.probability = 0.4
        
    def forward(self,dataIn):
        self.layer = []
        self.droppedNode = []
        self.setPrevIn(dataIn)
        for i in range(dataIn.shape[0]):
            for o in range(dataIn.shape[1]):        
                info = dataIn[i][o]
                droppedNode = np.random.rand(info.shape[0], info.shape[1]) > self.probability
                self.layer.append(droppedNode)
        return (np.array(self.layer).reshape(dataIn.shape) * dataIn) / self.probability
    def gradient(self) :
        prevIn = self.getPrevIn()
        return (np.array(self.layer).reshape(prevIn.shape)) / self.probability
class FullyConnectedLayer ( Layer ) :
    def __init__(self,sizeIn,sizeOut):
        super().__init__()
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.bias = np.random.uniform(low=-10**-4, high=10**-4, size=(sizeOut))
        self.weights = np.random.uniform(low=-10**-6, high=10**-4, size=(sizeIn,sizeOut))
        self.weights_grad = np.zeros(self.weights.shape)  
        self.b_grad = np.zeros(self.bias.shape)  
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

    def resetGradients(self):
        self.weights_grad = np.zeros(self.weights.shape)  
        self.b_grad = np.zeros(self.bias.shape)  

    def gradient(self) :
        return self.getWeights().T

    def backward(self,gradIn,eta=0.0001):
        gradOut = gradIn@self.gradient()
        x = gradIn.shape[0]
        djdw = (self.getPrevIn().T@gradIn)
        djdb = np.sum(gradIn,axis = 0)
        self.weights -= djdw*eta/x
        self.bias -= djdb*eta/x
        
        return gradOut
    
    def backwardNoGrad(self,gradIn):
        gradOut = gradIn@self.gradient()
        djdw = (self.getPrevIn().T@gradIn)
        djdb = np.sum(gradIn,axis = 0)
        self.weights_grad -= djdw
        self.b_grad -= djdb
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
        return gradIn@gradOut
    
class SigmoidLayer(Layer ):
    def __init__ (self ,dataIn=None):
        super().__init__()
        
    def forward(self ,dataIn): 
        super().setPrevIn(dataIn)
        dataIn = np.array(dataIn)
        if dataIn>=0:
            z = np.exp(-dataIn)
            prevOut = 1 / (1 + z)
        else:
            z = np.exp(dataIn)
            prevOut = z / (1 + z)
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
        return (1. * (pOut > 0))
    
class LeakyReLuLayer(Layer ):
    def __init__ (self ,dataIn=None, alpha = 0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        out = np.where((dataIn>0)==True, dataIn, dataIn*self.alpha)
        self.setPrevOut(out)
        return out
    
    def gradient(self):
        prevIn = self.getPrevIn()
        return np.where((prevIn>0)==True, 1, self.alpha)
    
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
        return (-2 * (y-yhat))  

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

class Conv():
    def __init__(self, inward, out, filter_size, stride=1, padding=0):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.out = out
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0
        weight_limit = 1/((inward*filter_size*filter_size)**(1/2))
        self.weight = np.random.rand(out,inward,filter_size,filter_size)*2*weight_limit-weight_limit
        self.bias = np.random.rand(out) * 2*weight_limit - weight_limit
        self._reset_gradients()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):   
        self.out_shape(x)
        p = self.padding
        if p != 0:
            pad_height = 2*p+x.shape[2] 
            pad_width = 2*p + x.shape[3] 
            padding_x = np.zeros((x.shape[0], x.shape[1], pad_height, pad_width))
            padding_x[:, :,p:-p, p:-p] = x
            x = padding_x

        
        self.flat_x = self._flatten_conv(x, self.stride)
      
        weight_flat = self.weight.reshape(self.weight.shape[0], -1) 
        flattenn = np.dot(self.flat_x, weight_flat.T) + self.bias
        conv = flattenn.T.reshape(*self.shape_out[1:], self.shape_out[0])

        return conv.transpose(3,0,1,2)

    def gradient(self, prev1):
        if prev1.shape != self.shape_out:
            prev1 = prev1.reshape(self.shape_out)
        prev1_flat = prev1.transpose(1,2,3,0).reshape(self.shape_out[1], -1)
        self.weights_grad += np.dot(prev1_flat, self.flat_x).reshape(self.weight.shape)
        self.b_grad += np.sum(prev1, axis=(0,2,3))
        return self._transposed_conv(prev1)

    def updateWeights(self, eta):
        self.weight -= eta*self.weights_grad
        self.bias -= eta*self.b_grad
        self._reset_gradients()
    def backwardADAMGrad(self, t, gradIn, beta1, beta2 , eta):
        epsilon = 1e-8
        djdw = gradIn
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.weight -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        
    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        epsilon = 1e-8
        djdw = self.weights_grad
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.weight -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        self._reset_gradients()
    
    def _reset_gradients(self):
        self.weights_grad = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def out_shape(self, x):
        h_out = int((x.shape[2] - self.filter_size + 2*self.padding)
                     //self.stride+1)
        w_out = int((x.shape[3] - self.filter_size + 2*self.padding)
                     //self.stride+1)

        self.shape_out = (x.shape[0], self.out, h_out, w_out)
        self.shape_in = x.shape

    def _flatten_conv(self, x, stride,h_out=None, w_out=None):

        if h_out == None and w_out == None:
            h_out, w_out = self.shape_out[2], self.shape_out[3]
        flatten_x = np.zeros((h_out*w_out*x.shape[0],x.shape[1]*self.filter_size*self.filter_size))
        h2 = h*stride+self.filter_size,w*stride
        h3 = w*stride+self.filter_size
        for h in range(0, h_out):
            flatten_h = h*w_out*x.shape[0]
            for w in range(0, w_out):
                flatten_w = w*x.shape[0]
                step_size = flatten_h+flatten_w
                flatten_x[step_size:step_size+x.shape[0],:] =x[:, :,h*stride:h2:h3].reshape(x.shape[0],-1)
        return flatten_x

    def _transposed_conv(self, x):
		
        if self.stride != 1:
            stride_h = self.stride*(x.shape[2])
            stride_w  = self.stride*(x.shape[3])
            x_stride = np.zeros((*x.shape[:2], stride_h, stride_w))
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x_stride[:, :, h*self.stride, w*self.stride] = x[:, :, h, w]
            x = x_stride
        padding = self.filter_size-1 - self.padding
        if padding == 0:
            pass
        elif padding < 0:
            padding = abs(padding)
            x = x[:, :, padding:-padding, padding:-padding]
        else:
            padding_x = np.zeros((x.shape[0], x.shape[1],
                              x.shape[2]+2*padding, x.shape[3]+2*padding))
            padding_x[:, :, padding:-padding, padding:-padding] = x
            x = padding_x
        x_flat = self._flatten_conv(x, 1, *self.shape_in[2:])

        weight_rot = np.flip(self.weight, (2,3))
        weights1flatten = weight_rot.transpose(1,0,2,3).reshape(self.weight.shape[1],-1)
        trans_flattenn = np.dot(x_flat, weights1flatten.T)
        trans_conv = trans_flattenn.T.reshape(*self.shape_in[1:],self.shape_in[0])
        return trans_conv.transpose(3,0,1,2)

class TransposedConv():
    def __init__(self, inward, out, filter_size, stride=1, padding=0):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.out = out
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0
        weight_limit = 1 / ((inward*filter_size*filter_size)**(1/2))
        self.weight = np.random.rand(out, inward, filter_size, filter_size)* 2*weight_limit - weight_limit
        self.bias = np.random.rand(out) * 2*weight_limit - weight_limit
        self._reset_gradients()
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.shape_in = x.shape
        if self.stride != 1:
            stride_h = self.stride*(x.shape[2])
            stride_w  = self.stride*(x.shape[3])
            x_stride = np.zeros((*x.shape[:2], stride_h, stride_w))
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x_stride[:, :, h*self.stride, w*self.stride] = x[:, :, h, w]
            x = x_stride[:,:,:-self.stride+1,:-self.stride+1]
        padding = self.filter_size-1 - self.padding
        if padding == 0:
            pass
        elif padding < 0:
            padding = abs(padding)
            x = x[:, :, padding:-padding, padding:-padding]
        else:
            padding_x = np.zeros((x.shape[0], x.shape[1], x.shape[2]+2*padding, x.shape[3]+2*padding))
            padding_x[:, :, padding:-padding, padding:-padding] = x
            x = padding_x

        self.out_shape(x)


        self.x_flat = self._flatten_conv(x, 1)
        weight_rot = np.flip(self.weight, (2,3))
        weight_flat = weight_rot.reshape(self.weight.shape[0], -1)
        trans_flattenn = np.dot(self.x_flat, weight_flat.T) + self.bias
        trans_conv = trans_flattenn.T.reshape(*self.shape_out[1:], self.shape_out[0])
        trans_conv = trans_conv.transpose(3,0,1,2)

        return trans_conv

    def gradient(self, prev1):
 
        if prev1.shape != self.shape_out:
            prev1 = prev1.reshape(self.shape_out)

        prev1_flat = prev1.transpose(1,2,3,0).reshape(self.shape_out[1], -1)


        weights_grad = np.dot(prev1_flat, self.x_flat).reshape(self.weight.shape)
        self.weights_grad += np.flip(weights_grad, (2,3))
        self.b_grad += np.sum(prev1, axis=(0,2,3))
        next_grad = self._conv(prev1)

        return next_grad 

    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        epsilon = 1e-8
        djdw = self.weights_grad
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.weight -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        self._reset_gradients()
        
    def backwardADAMGrad(self, t, gradIn, beta1, beta2 , eta):
        epsilon = 1e-8
        djdw = gradIn
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.weight -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
    
    def updateWeights(self, eta):
        self.weight -= eta*self.weights_grad
        self.bias -= eta*self.b_grad
        self._reset_gradients()
    
    def _reset_gradients(self):
        self.weights_grad = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def out_shape(self, x):
        h_out = max(0, x.shape[2] - self.filter_size + 1)
        w_out = max(0, x.shape[3] - self.filter_size + 1)

        self.shape_out = (x.shape[0], self.out, h_out, w_out)

    def _flatten_conv(self, x, stride,h_out=None, w_out=None):  

        if h_out == None and w_out == None:
            h_out, w_out = self.shape_out[2], self.shape_out[3]
        h1 = h*stride+self.filter_size,w*stride
        h2 = w*stride+self.filter_size
        flatten_x = np.zeros((h_out*w_out*x.shape[0],x.shape[1]*self.filter_size*self.filter_size))
        for h in range(0, h_out):
            flatten_h = h*w_out*x.shape[0]
            for w in range(0, w_out):
                flatten_w = w*x.shape[0]
                step_size = flatten_h+flatten_w

                flatten_x[step_size:step_size+x.shape[0],:] =x[:, :,h*stride:h1:h2].reshape(x.shape[0],-1)
        return flatten_x

    def _conv(self, prev1):
        if self.padding != 0:
            pad_height = prev1.shape[2] + 2*self.padding
            pad_width = prev1.shape[3] + 2*self.padding
            grad_pad = np.zeros((prev1.shape[0], prev1.shape[1],pad_height, pad_width))
            grad_pad[:, :,
                  self.padding:-self.padding,
                  self.padding:-self.padding] = prev1
            prev1 = grad_pad

        flat_grad = self._flatten_conv(prev1, self.stride, *self.shape_in[2:])
        weight_flat = self.weight.transpose(1,0,2,3).reshape(self.weight.shape[1], -1)
        flattenn = np.dot(flat_grad, weight_flat.T)
        conv = flattenn.T.reshape(*self.shape_in[1:], self.shape_in[0])
        conv = conv.transpose(3,0,1,2)

        return conv


    
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
                    print(123)
                j += self.size
            i += self.size
        return grad
    
    def backward(self, gradIn):
        prevIn = self.getPrevIn()
        grad = np.zeros(prevIn.shape)
        i = 0
        j = 0
        indxI = 1
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

def randomUniformMatrix(l=5):
    return np.random.uniform(low=-10**-4, high=10**-4, size=(l,l))

def space_to_depth(x, block_size):
    x = np.asarray(x)
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size,
                         reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z


def resizeImage(img, size):
    scale_percent = size 
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

def calculate_psnr(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def generateFake(h, w):
    X = np.random.rand(h * w)
    X = X.reshape((h, w))
    return X

def interleave(inputMatrix):
    tempMatrix = np.ndarray.flatten(inputMatrix)
    arr1 = np.zeros(tempMatrix.size)
    arr_tuple = (tempMatrix, arr1)
    interleaved = np.vstack(arr_tuple).reshape((-1,), order='F')
    
    zeroCol = np.reshape(interleaved, (inputMatrix.shape[0],inputMatrix.shape[1]*2))
    
    a=np.zeros((zeroCol.shape[0]*2,zeroCol.shape[1])) 
    a[::2] = zeroCol
    return a




class Discriminator():
    def __init__ (self):  
        self.L1 = Conv(1,1,3,1,1)
        self.L2 = LeakyReLuLayer()
        self.L3 = FullyConnectedLayer(12060, 1)
        self.L4 = SigmoidLayer()

    def getWeights(self):
        return (self.L1.weight, self.L3.getWeights())
        
    def resetDiscriminator(self):
        self.L1 = Conv(1,1,3,1,1)
        self.L2 = LeakyReLuLayer()
        self.L3 = FullyConnectedLayer(12060, 1)
        self.L4 = SigmoidLayer()

    def discriminate(self, inImg):
        l1Out = self.L1.forward(inImg)
        l2Out = self.L2.forward(l1Out)
        l2OutFlat = np.ndarray.flatten(l2Out)
        l2OutFlat = np.array([l2OutFlat])
        l3Out = self.L3.forward(l2OutFlat) 
        l4Out = self.L4.forward(l3Out)
        return l4Out
    
    def discriminateTrain(self, t, gradientLoss):
        l4Out = self.L4.gradient() * gradientLoss
        l3Out = self.L3.backwardADAM(t, l4Out, 0.99, 0.999 , 0.001)
        l3Out = l3Out.reshape((1, 1, 134, 90))
        l2gradient = self.L2.gradient()
        l2Out = l3Out*l2gradient
        self.L1.gradient(l2Out)
        updateGrad = self.L1.weights_grad
        self.L1.backwardADAM(t, updateGrad, 0.99, 0.999 , 0.001)  
    
    def discriminateNotTrain(self, gradientLoss):
        l4Out = self.L4.gradient() * gradientLoss
        l3Out = l4Out@self.L3.gradient()
        l3Out = l3Out.reshape((1, 1, 134, 90))
        l2gradient = self.L2.gradient()
        l2Out = l3Out*l2gradient
        out = self.L1.gradient(l2Out)
        self.L1._reset_gradients()
        return out      
        
class Generator():
    def __init__ (self): 
        self.L3 = TransposedConv(1,1,2,2,0)
        self.L4 = LeakyReLuLayer()
        self.L5 = Conv(1,1,3,1,1)  
        self.L6 = LeakyReLuLayer()

    def resetGenerator(self):
        self.L3 = TransposedConv(1,1,2,2,0)
        self.L4 = LeakyReLuLayer()
        self.L5 = Conv(1,1,3,1,1) 
        self.L6 = LeakyReLuLayer()
        
    def generate(self, inImg):
        l3Out = self.L3.forward(inImg)
        l4Out = self.L4.forward(l3Out)
        l5Out = self.L5.forward(l4Out)
        l6Out = self.L6.forward(l5Out)
        return l6Out


    
    def generateTrainMSE(self, t, gradientLoss):
        l6Out = self.L6.gradient()
        gradientLoss = l6Out * gradientLoss
        l5Out = self.L5.gradient(gradientLoss)
        updateGrad = self.L5.weights_grad
        self.L5.backwardADAM(t, updateGrad, 0.99, 0.999 , 0.001)
        
        l4gradient = self.L4.gradient()
        l4Out = l5Out*l4gradient
        
        l3Out = self.L3.gradient(l4Out)
        updateGrad = self.L3.weights_grad
        self.L3.backwardADAM(t, updateGrad, 0.99, 0.999 , 0.001)

               
def generateImg(inImg):
    inImg = np.array([[inImg]])
    genImg = generator.generate(inImg) 
    return genImg
    
def updateGeneratorMSE(t, loss):
    generator.generateTrainMSE(t, loss)

def updateGeneratorGAN(t, loss):
    generator.generateTrainGAN(t, loss)
    
def discriminateImg(inImg):
    inImg = np.array([[inImg]])
    yhat = discriminator.discriminate(inImg)
    return yhat

def updateDiscriminator(t, loss):
    discriminator.discriminateTrain(t, loss)

def lossDiscriminator(loss):
    weights = discriminator.getWeights()
    gradient =  discriminator.discriminateNotTrain(loss)
#    if weights == discriminator.getWeights():
#        print("cool")
    return gradient

path1 = "SampleMoviePosters/"

listing = os.listdir(path1)    


path2 = "ValidatePosters/"

validation = os.listdir(path2)    


im = None

discrimLoss = []

discriminator = Discriminator()
generator = Generator()


LL = LogLoss()
MSE = LeastSquares()

resume = False

if not resume:
    with open('discriminator_pkl', 'wb') as files:
        pickle.dump(discriminator, files)
    
    with open('generator_pkl', 'wb') as files:
        pickle.dump(generator, files)

else: 
    with open('discriminator_pkl' , 'rb') as f:
        discriminator = pickle.load(f)
    with open('generator_pkl' , 'rb') as f:
        generator = pickle.load(f)    
    

    
epoch = 2000

isDiscriminator = True

for i in range(epoch):
    print(i)
    disLoss = 0

    random.shuffle(listing)
    for itm in listing:
        if itm == '.DS_Store':
            continue
        real = True
     
        im = cv2.imread(path1 + itm,0) 
        im = resizeImage(im, 50)
        image = im[:,:-1]
        image = np.array(image, dtype=np.float64)
        
        imgY = np.ndarray.flatten(image) 

        if random.getrandbits(1) or not isDiscriminator:
            real = False            
            
            startImage = resizeImage(image, 50)
            image = generateImg(startImage)

            image = np.array(image[0][0], dtype=np.float64)

        
        yhat = discriminateImg(image)
        
        y = np.array([0])
        if real:
            y = np.array([1])

        disLoss += LL.eval(y,yhat[0])

        gradient = LL.gradient(y,yhat[0])

        if isDiscriminator:
            discriminator.discriminateTrain(i+1, gradient)
        
        else:
            disGrad = discriminator.discriminateNotTrain(gradient)
            disGrad = disGrad * 10**-3
            
            grad = MSE.gradient(imgY, np.ndarray.flatten(image))
            grad = grad.reshape((134, 90))
            generator.generateTrainMSE(i+1, np.array([[grad]]))
            generator.generateTrainMSE(i+1, np.array(disGrad))
            

    discriminatorLoss = disLoss/len(listing)
    
    if not isDiscriminator:
        for itm in validation:
            if itm == '.DS_Store':
                continue
            real = True
            
            im = cv2.imread(path2 + itm,0) 
            im = resizeImage(im, 50)
            image = im[:,:-1]

            startImage = resizeImage(image, 50)
            imgY = generateImg(startImage)
            cv2.imwrite('end_' +  "{0:0=4d}".format(i) + "_" + str(itm).replace(".jpg","") + ".png" , imgY[0][0]) 
        
        
    if not isDiscriminator:
        print(discriminatorLoss)
        if discriminatorLoss >= 20:
            print("swapping to Discriminator")
            isDiscriminator = not isDiscriminator
            dis = Discriminator()
            
    if isDiscriminator:
        print(discriminatorLoss)
        if discriminatorLoss < 20:   
            print("swapping to Generator")
            isDiscriminator = not isDiscriminator
    
    
    discrimLoss.append(disLoss/len(listing))    
        



plt.plot(range(len(discrimLoss)), discrimLoss, label="upscaleLoss")   
plt.title("")
plt.xlabel("epochs")
plt.ylabel("Log Loss Avg")
figure = plt.gcf()
figure.set_size_inches(6, 8)
plt.legend()
#plt.savefig("MNIST.png", dpi=200)
plt.show()  




