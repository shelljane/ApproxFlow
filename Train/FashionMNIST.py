import random
import h5py
import numpy as np

import tensorflow as tf

import Preproc
import Layer
import Net

def loadHDF5():
    with h5py.File('FashionMNIST.h5', 'r') as f:
        dataTrain   = np.expand_dims(np.array(f['Train']['images'])[:, :, :, 0], axis=-1)
        labelsTrain = np.array(f['Train']['labels']).reshape([-1])
        dataTest    = np.expand_dims(np.array(f['Test']['images'])[:, :, :, 0], axis=-1)
        labelsTest  = np.array(f['Test']['labels']).reshape([-1])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def preproc(images, size): 
    results = np.ndarray([images.shape[0]]+size, np.uint8)
    for idx in range(images.shape[0]): 
        distorted     = Preproc.centerCrop(images[idx], size)
        results[idx]  = distorted
    
    return results

def allData(preprocSize=[28, 28, 1]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    return preproc(data, preprocSize), labels, invertedIdx


def generators(BatchSize, preprocSize=[28, 28, 1]):
    ''' generators for multi-let
    Args:
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5()
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[0], shuffle=True)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTrain[indexAnchor]
            labelAnchor = labelsTrain[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[0], shuffle=False)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTest[indexAnchor]
            labelAnchor = labelsTest[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
    
    def preprocTrain(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            #distorted     = Preproc.randomFlipH(images[idx])
            distorted     = Preproc.randomShift(images[idx], rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            #distorted     = Preproc.randomRotate(images[idx], rng=30)
            distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted = images[idx]
            distorted     = Preproc.centerCrop(distorted, size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
        
    return genTrainBatch(BatchSize), genTestBatch(BatchSize)

HParamMNIST = {'NumGPU': 1, 
                 'BatchSize': 200, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 1e-5, 
                 'WeightDecay': 1e-4, 
                 'ValidateAfter': 300,
                 'LRDecayAfter': 3000,
                 'LRDecayRate': 0.1,
                 'TestSteps': 50,
                 'TotalSteps': 3000}

HParamMNIST_Quant = {'NumGPU': 1, 
                    'BatchSize': 200, 
                    'LearningRate': 1e-4, 
                    'MinLearningRate': 1e-5, 
                    'WeightDecay': 1e-4, 
                    'ValidateAfter': 300,
                    'LRDecayAfter': 3000,
                    'LRDecayRate': 0.1,
                    'TestSteps': 50,
                    'TotalSteps': 3000}

if __name__ == '__main__':
    batchTrain, batchTest = generators(BatchSize=HParamMNIST['BatchSize'], preprocSize=[28, 28, 1])
    
    net = Net.Net4Classify(inputShape=[28, 28, 1], numClasses=10, \
                           body=Net.LeNetBigBody, HParam=HParamMNIST, name='Net4Classify')
    net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netMNIST.ckpt')
    
    net = Net.Net4Quant(inputShape=[28, 28, 1], numClasses=10, \
                        body=Net.LeNetBigBody_Quant, pretrained=net, HParam=HParamMNIST_Quant, name='Net4Quant')
    net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netMNIST_Quant.ckpt')
    
    net = Net.Net4Approx(inputShape=[28, 28, 1], numClasses=10, \
                         body=Net.LeNetBigBody_Quant, pretrained=net, HParam=HParamMNIST_Quant, name='Net4Approx')
    net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netMNIST_Approx.ckpt')
