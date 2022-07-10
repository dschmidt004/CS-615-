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
        #sg = self.gradient()
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
        return gradIn@gradOut
    
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
        self.pad = 0
        self.stride = 1
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0

    def forward(self ,dataIn, pad, stride): 


        self.pad = pad
        self.stride = stride
        if pad > 0:
            dataIn = np.pad(dataIn, pad, self.pad_with)
        super().setPrevIn(dataIn)
        if np.isnan(np.sum(pad)):
            sys.exit()
        if np.isnan(np.sum(dataIn)):
            sys.exit()
        out = self.convo(dataIn, self.kernel, stride)
        super().setPrevOut(out)
        return out

    def gradient(self, gradIn): 
        prevIn = super().getPrevIn()
        return self.convo(prevIn, gradIn, self.stride)

    def pad_with(self, vector, pad_width, iaxis, kwargs):

        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
    
    def convo(self, dataIn, kernel, stride):
        out = np.zeros((int((dataIn.shape[0]-kernel.shape[0])/stride + 1),int((dataIn.shape[1]-kernel.shape[1])/stride +1)) )
        i = 0
        j = 0
        while i+kernel.shape[0] <= dataIn.shape[0]:
            j = 0
            while j+kernel.shape[1] <= dataIn.shape[1]:
                subArry = dataIn[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                iS = int(i/stride)
                jS = int(j/stride)
                out[iS][jS] = np.sum(subArry*kernel)

                j+=stride
            i+=stride
        return out

    def backward(self, gradIn, eta=0.0001):
        prevIn = super().getPrevIn()
        temp = self.kernel
        pad = self.pad
        if np.isnan(np.sum(temp)):
            sys.exit()
        if np.isnan(np.sum(pad)):
            sys.exit()
        if np.isnan(np.sum(prevIn)):
            sys.exit()
        if np.isnan(np.sum(gradIn)):
            sys.exit()

        if pad > 0:
            prevIn = prevIn[pad:-pad,pad:-pad]
        djdw = self.convo(prevIn, gradIn, self.stride)
        self.kernel -= eta*djdw
        gradIn = np.pad(gradIn, int(self.kernel.shape[0]/2), self.pad_with)
        return self.convo(gradIn, np.flip(temp), self.stride)

    
    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        prevIn = super().getPrevIn()
        pad = self.pad
        stride = self.stride
        if pad > 0:
            prevIn = prevIn[pad:-pad,pad:-pad]
        temp = self.kernel.copy()
        djdw = self.convo(prevIn, gradIn, stride)
        epsilon = 1e-8
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.kernel -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        gradIn = np.pad(gradIn, int(self.kernel.shape[0]/2), self.pad_with)
        return self.convo(gradIn, np.flip(temp), self.stride)


class Conv():
    def __init__(self, in_chan, out_chan, filter_size, stride=1, dem_pad=0):
        self.filter_size = filter_size
        self.stride = stride
        self.dem_pad = dem_pad
        self.out_chan = out_chan

        weight_limit = 1 / ((in_chan*filter_size*filter_size)**(1/2))
        self.weight = np.random.rand(out_chan, in_chan, filter_size, filter_size)\
                      * 2*weight_limit - weight_limit
        #self.bias = np.zeros(out_chan)
        self.bias = np.random.rand(out_chan) * 2*weight_limit - weight_limit

        self._reset_gradients()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):   
        self._calculate_out_shape(x)
        if self.dem_pad != 0:
            pad_h = x.shape[2] + 2*self.dem_pad
            pad_w = x.shape[3] + 2*self.dem_pad
            x_pad = np.zeros((x.shape[0], x.shape[1], pad_h, pad_w))
            x_pad[:, :,
                  self.dem_pad:-self.dem_pad,
                  self.dem_pad:-self.dem_pad] = x
            x = x_pad


        self.flat_x = self._flattenconvolu1(x, self.stride)
        weight_flat = self.weight.reshape(self.weight.shape[0], -1)
        conv_flat = np.dot(self.flat_x, weight_flat.T) + self.bias
        conv = conv_flat.T.reshape(*self.shape_out[1:], self.shape_out[0])
        conv = conv.transpose(3,0,1,2)

        return conv

    def gradient(self, prev_grad):
        if prev_grad.shape != self.shape_out:
            prev_grad = prev_grad.reshape(self.shape_out)
        grad_flat  = prev_grad.transpose(1,2,3,0).reshape(self.shape_out[1], -1)
        self.gradients_weights += np.dot(grad_flat , self.flat_x).reshape(self.weight.shape)
        self.b_grad += np.sum(prev_grad, axis=(0,2,3))
      

        return self._transposedconvolu1(prev_grad)

    def _reset_gradients(self):
        self.gradients_weights = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def _calculate_out_shape(self, x):
        h_out = int((x.shape[2] - self.filter_size + 2*self.dem_pad)
                     //self.stride+1)
        w_out = int((x.shape[3] - self.filter_size + 2*self.dem_pad)
                     //self.stride+1)

        self.shape_out = (x.shape[0], self.out_chan, h_out, w_out)
        self.shape_in = x.shape

    def _flattenconvolu1(self, x, stride,h_out=None, w_out=None):

        if h_out == None and w_out == None:
            h_out, w_out = self.shape_out[2], self.shape_out[3]
        flatten1 = np.zeros((h_out*w_out*x.shape[0],x.shape[1]*self.filter_size*self.filter_size))

        for h in range(0, h_out):
            flatten_h = h*w_out*x.shape[0]
            for w in range(0, w_out):
                flatten_w = w*x.shape[0]
                step_size = flatten_h+flatten_w
                flatten1[step_size:step_size+x.shape[0],:] =\
                        x[:, :,
                          h*stride:h*stride+self.filter_size,
                          w*stride:w*stride+self.filter_size].reshape(
                                                                x.shape[0],-1)
        return flatten1

    def _transposedconvolu1(self, x):  
        if self.stride != 1:
            stride_h = self.stride*(x.shape[2])
            stride_w  = self.stride*(x.shape[3])
            x_stride = np.zeros((*x.shape[:2], stride_h, stride_w))
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x_stride[:, :, h*self.stride, w*self.stride] = x[:, :, h, w]
            x = x_stride
        dem_pad = self.filter_size-1 - self.dem_pad
        if dem_pad == 0:
            pass
        elif dem_pad < 0:
            dem_pad = abs(dem_pad)
            x = x[:, :, dem_pad:-dem_pad, dem_pad:-dem_pad]
        else:
            x_pad = np.zeros((x.shape[0], x.shape[1],
                              x.shape[2]+2*dem_pad, x.shape[3]+2*dem_pad))
            x_pad[:, :, dem_pad:-dem_pad, dem_pad:-dem_pad] = x
            x = x_pad


        x_flat = self._flattenconvolu1(x, 1, *self.shape_in[2:])
        weight_rot = np.flip(self.weight, (2,3))
        weight_rot_flat = weight_rot.transpose(1,0,2,3).reshape(self.weight.shape[1],-1)
        transconvolu1_flat = np.dot(x_flat, weight_rot_flat.T)
        transconvolu1 = transconvolu1_flat.T.reshape(*self.shape_in[1:],self.shape_in[0])
        return transconvolu1.transpose(3,0,1,2)

class TransposedConv():
    def __init__(self, in_chan, out_chan, filter_size, stride=1, dem_pad=0):
        self.filter_size = filter_size
        self.stride = stride
        self.dem_pad = dem_pad
        self.out_chan = out_chan
        self.momentumDW, self.velocityDW = 0, 0
        self.momentumDB, self.velocityDB = 0, 0
        weight_limit = 1 / ((in_chan*filter_size*filter_size)**(1/2))
        self.weight = np.random.rand(out_chan, in_chan, filter_size, filter_size)* 2*weight_limit - weight_limit
        #self.bias = np.zeros(out_chan)
        self.bias = np.random.rand(out_chan) * 2*weight_limit - weight_limit
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

        dem_pad = self.filter_size-1 - self.dem_pad
        
        if dem_pad < 0:
            dem_pad = abs(dem_pad)
            x = x[:, :, dem_pad:-dem_pad, dem_pad:-dem_pad]
        elif dem_pad == 0:
            pass
        else:
            x_pad = np.zeros((x.shape[0], x.shape[1], x.shape[2]+2*dem_pad, x.shape[3]+2*dem_pad))
            x_pad[:, :, dem_pad:-dem_pad, dem_pad:-dem_pad] = x
            x = x_pad
        self._calculate_out_shape(x)
        self.x_flat = self._flattenconvolu1(x, 1)
        weight_rot = np.flip(self.weight, (2,3))
        weight_flat = weight_rot.reshape(self.weight.shape[0], -1)
        transconvolu1_flat = np.dot(self.x_flat, weight_flat.T) + self.bias
        transconvolu1 = transconvolu1_flat.T.reshape(*self.shape_out[1:], self.shape_out[0])
        return transconvolu1.transpose(3,0,1,2)

    def gradient(self, prev_grad):
        if prev_grad.shape != self.shape_out:
            prev_grad = prev_grad.reshape(self.shape_out)

        grad_flat  = prev_grad.transpose(1,2,3,0).reshape(self.shape_out[1], -1)
        w_grad = np.dot(grad_flat , self.x_flat).reshape(self.weight.shape)
        self.gradients_weights += np.flip(w_grad, (2,3))
        self.b_grad += np.sum(prev_grad, axis=(0,2,3))
        return self.convolu1(prev_grad)

    def backwardADAM(self, t, gradIn, beta1, beta2 , eta):
        nextGrad = self.gradient(gradIn)
        epsilon = 1e-8
        djdw = self.gradients_weights
        self.momentumDW = beta1*self.momentumDW + (1-beta1)*djdw
        self.velocityDW = beta2*self.velocityDW + (1-beta2)*(djdw*djdw)
        SNum = self.momentumDW/(1-beta1**t)
        RDenom = self.velocityDW/(1-beta2**t)
        self.weight -= eta*(SNum/(np.sqrt(RDenom)+ epsilon))
        self._reset_gradients()
        return nextGrad
    
    def _reset_gradients(self):
        self.gradients_weights = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def _calculate_out_shape(self, x):
        h_out = max(0, x.shape[2] - self.filter_size + 1)
        w_out = max(0, x.shape[3] - self.filter_size + 1)

        self.shape_out = (x.shape[0], self.out_chan, h_out, w_out)

    def _flattenconvolu1(self, x, stride,
                      h_out=None, w_out=None):  

        if h_out == None and w_out == None:
            h_out, w_out = self.shape_out[2], self.shape_out[3]
        flatten1 = np.zeros((h_out*w_out*x.shape[0],x.shape[1]*self.filter_size*self.filter_size))
        for h in range(0, h_out):
            flatten_h = h*w_out*x.shape[0]
            for w in range(0, w_out):
                flatten_w = w*x.shape[0]
                step_size = flatten_h+flatten_w

                flatten1[step_size:step_size+x.shape[0],:] =x[:, :,h*stride:h*stride+self.filter_size,w*stride:w*stride+self.filter_size].reshape(x.shape[0],-1)
        return flatten1

    def convolu1(self, prev_grad):
        if self.dem_pad != 0:
            pad_h = 2*self.dem_pad + prev_grad.shape[2] 
            pad_w = 2*self.dem_pad + prev_grad.shape[3] 
            grad_pad = np.zeros((prev_grad.shape[0], prev_grad.shape[1],pad_h, pad_w))
            grad_pad[:, :,self.dem_pad:-self.dem_pad,self.dem_pad:-self.dem_pad] = prev_grad
            prev_grad = grad_pad

        flat_grad = self._flattenconvolu1(prev_grad, self.stride, *self.shape_in[2:])
        weight_flat = self.weight.transpose(1,0,2,3).reshape(self.weight.shape[1], -1)
        conv_flat = np.dot(flat_grad, weight_flat.T)
        conv = conv_flat.T.reshape(*self.shape_in[1:], self.shape_in[0])
        return conv.transpose(3,0,1,2)

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
                    print(e)
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


def resizeImage(img):
    scale_percent = 50 # percent of original size
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

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
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


def interleave(inputMatrix):
    tempMatrix = np.ndarray.flatten(inputMatrix)
    arr1 = np.zeros(tempMatrix.size)
    arr_tuple = (tempMatrix, arr1)
    interleaved = np.vstack(arr_tuple).reshape((-1,), order='F')
    
    zeroCol = np.reshape(interleaved, (inputMatrix.shape[0],inputMatrix.shape[1]*2))
    
    a=np.zeros((zeroCol.shape[0]*2,zeroCol.shape[1])) 
    a[::2] = zeroCol
    return a

path1 = "SampleMoviePosters/"

listing = os.listdir(path1)    


path2 = "ValidatePosters/"

validation = os.listdir(path2)    

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

#L1up = ConvLayer2DTransposed()
#L1up.kernel =  randomUniformMatrix(2)

L1up = TransposedConv(1,1,2,2,0)

L5 = LeastSquares()
lossArry = []
validArry = []
upLossArry = []
epoch = 2000
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

        startImage = resizeImage(im)

        y = np.ndarray.flatten(im)

        inImg = IPL.forward(startImage)

        #inImg = interleave(inImg)
        inImg = np.array([[inImg]])
        out1 = L1up.forward(inImg)
        
        out2 = np.ndarray.flatten(out1[0][0])
        
        yhat = np.ndarray.flatten(out2)
        
        loss += np.sum(L5.eval(y, yhat))
        
        lossGrad = np.array(L5.gradient(y, yhat))
        
        lossGrad = np.reshape(lossGrad, (134, 90)) 
        
        lossGrad = np.array([[lossGrad]])
        
        L1up.backwardADAM(i+1, lossGrad, 0.99,0.999,0.001) 

        
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
        y = np.ndarray.flatten(im)

        startImage = resizeImage(im)

        y = np.ndarray.flatten(im)

        inImg = IPL.forward(startImage)

        #inImg = interleave(inImg)
        inImg = np.array([[inImg]])
        out1 = L1up.forward(inImg)
        out2 = np.ndarray.flatten(out1[0][0])
        yhat = np.ndarray.flatten(out2)
        
        loss += np.sum(L5.eval(y, yhat))
        
        cv2.imwrite('start_' + str(i) + "_" + str(itm) , startImage)
        cv2.imwrite('target_' + str(itm) , im)
        cv2.imwrite('end_' + str(i) + "_" + str(itm) , out1[0][0])   

    validArry.append(loss/count)

plt.plot(range(len(lossArry)), lossArry, label="Train")
plt.plot(range(len(validArry)), validArry, label="Valid")   
#plt.plot(range(len(upLossArry)), upLossArry, label="upscaleLoss")   
plt.title("")
plt.xlabel("epochs")
plt.ylabel("SME Loss Avg")
figure = plt.gcf()
figure.set_size_inches(6, 8)
plt.legend()
#plt.savefig("MNIST.png", dpi=200)
plt.show()  



