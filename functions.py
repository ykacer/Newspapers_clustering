import os
import cv2
import numpy as np


def descr_rgb(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1]*patchs.shape[2],patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
         patchs_preprocessed[:,n] = patchs[:,:,:,n].flatten()
    return patchs_preprocessed

def descr_hsv(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1]*patchs.shape[2],patchs.shape[3]))
    #bins = np.arange(256)
    #patchs_preprocessed = np.zeros((3*(bins.size-1),patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
         patch_hsv = cv2.cvtColor(patchs[:,:,::-1,n],cv2.COLOR_BGR2HSV)
         #patchs_preprocessed[:,n] = np.hstack((np.histogram(patch_hsv[:,:,0],bins)[0] ,np.histogram(patch_hsv[:,:,1],bins)[0], np.histogram(patch_hsv[:,:,2],bins)[0]))
         patchs_preprocessed[:,n] = patch_hsv.flatten()
    return patchs_preprocessed

def descr_grad(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1],patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
        patchs_preprocessed[:,n] = cv2.cvtColor(cv2.Laplacian(np.squeeze(patchs[:,:,:,n].astype(np.uint8)),cv2.CV_8U),cv2.COLOR_RGB2GRAY).flatten()
    return patchs_preprocessed

def descr_hog(patchs):
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    winSize = (patchs.shape[0],patchs.shape[1])
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    feature_size = (winSize[0]-(blockSize[0]-blockStride[0]))/blockStride[0]*(winSize[1]-(blockSize[1]-blockStride[1]))/blockStride[1]*blockSize[0]/cellSize[0]*blockSize[1]/cellSize[1]*nbins
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    patchs_preprocessed = np.zeros((feature_size,patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
        patchs_preprocessed[:,n] = hog.compute(np.squeeze(patchs[:,:,:,n].astype(np.uint8))).flatten()
    return patchs_preprocessed

