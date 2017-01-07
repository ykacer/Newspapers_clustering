import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy import misc
from utils import *
from functions import *
import sys

filename = sys.argv[1]

# parameters #
descr = ['hog','hsv']
decomposition = 'kmeans'
flatten_or_not = False
color_mapping = {0:[255,0,0],1:[0,0,255],2:[255,255,255]}
resizing_factor = 4
roi_size_x = 80/resizing_factor
roi_size_y = 80/resizing_factor
overlap_x = 0/resizing_factor
overlap_y = 0/resizing_factor
#############

def main():
    image = cv2.imread(filename)

    ### resize image for faster computation
    image = cv2.resize(image,(image.shape[1]/resizing_factor,image.shape[0]/resizing_factor))
    print "processing "+filename+"..."

    ### binarize image to find patchs corresponding to class 2 (background)
    binary_thresh = 200
    gray = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(3,3),0)
    ret,binary = cv2.threshold(gray,binary_thresh,255,cv2.THRESH_BINARY)
    #ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w,c = image.shape

    ### cut image into patchs, keep only patchs which are not class 2.
    patchs,list_patchs_x,list_patchs_y = cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not,binary)

    ### extract features of each patch
    nb_patchs = patchs.shape[-1]
    patchs_preprocessed = np.array([]).reshape(0,nb_patchs)
    for d in descr:
        if d == 'grad':
            patchs_preprocessed = np.concatenate((patchs_preprocessed,descr_grad(patchs)),axis=0)
        if d == 'hog':
            patchs_preprocessed = np.concatenate((patchs_preprocessed,descr_hog(patchs)),axis=0)
        if d == 'rbg':
            patchs_preprocessed = np.concatenate((patchs_preprocessed,descr_rgb(patchs)),axis=0)
        if d == 'hsv':
            patchs_preprocessed = np.concatenate((patchs_preprocessed,descr_hsv(patchs)),axis=0)

    # patchs clustering
    if decomposition == 'nmf':
        ### compute NMF classification to find class for left patchs : class 0 (illustration) or class 1 (texte) 
        nmf = NMF(n_components=3)
        patchs_transformed = nmf.fit_transform(patchs_preprocessed.transpose())
        ind = np.argmax(patchs_transformed,axis=1)
    elif decomposition == 'kmeans':
        ### compute KMEANS classification to find class for left patchs : class 0 (illustration) or class 1 (texte) 
        km = KMeans(n_clusters=2)#3)
        ind = km.fit_predict(patchs_preprocessed.transpose())
    else: 
        print decomposition + " : decomposition unknown"

    ### patchs are now divide into categories, give class 1 (illustration) for category with greater proportion of "white" (i.e. > binary_thresh).
    p0 = 1.0*np.sum(np.squeeze(0.1140*patchs[:,:,0,ind==0]+0.5870*patchs[:,:,1,ind==0]+0.2989*patchs[:,:,2,ind==0]).flatten()>binary_thresh)/patchs.size
    p1 = 1.0*np.sum(np.squeeze(0.1140*patchs[:,:,0,ind==1]+0.5870*patchs[:,:,1,ind==1]+0.2989*patchs[:,:,2,ind==1]).flatten()>binary_thresh)/patchs.size
    sorted_labels = np.argsort([p0,p1])
    color_mapping[sorted_labels[0]] = [255,0,0] # red for class 0 (illustration) 
    color_mapping[sorted_labels[1]] = [0,0,255] # blue for class 1 (texte) 

    ### compute prediction (class 0,1, or 2) for each pixel
    image_result = 255*np.ones((h,w,c))
    for n,(j,i) in enumerate(izip(list_patchs_y,list_patchs_x)):
        image_result[j:j+roi_size_y,i:i+roi_size_x,:] = color_mapping[ind[n]]

    # post-processing
    image_result = post_processing(image_result)
    
    res_name = "_res"
    for d in descr:
        res_name = res_name + "_" + d
    res_name = res_name + '_' + decomposition
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(image[:,:,::-1])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    f.add_subplot(1,3,2)
    plt.imshow(binary,cmap=plt.get_cmap('gray'));
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    f.add_subplot(1,3,3)
    plt.imshow(image_result.astype(np.uint8))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    #plt.show()
    f.savefig(filename[:-4]+res_name+'_all.jpg', dpi=f.dpi)
    f.clf()
    print "saving result : ",filename[:-4]+res_name+'.jpg'
    misc.imsave(filename[:-4]+res_name+'.jpg',cv2.resize(image_result,(w*resizing_factor,h*resizing_factor)).astype(np.uint8))


if __name__ == '__main__':
    main()


