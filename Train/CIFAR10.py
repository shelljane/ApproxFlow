import random
import h5py
import numpy as np

import tensorflow as tf

import Preproc
import Layer
import Net

def loadHDF5():
    with h5py.File('CIFAR10.h5', 'r') as f:
        dataTrain   = np.array(f['Train']['images'])
        labelsTrain = np.array(f['Train']['labels'])
        dataTest    = np.array(f['Test']['images'])
        labelsTest  = np.array(f['Test']['labels'])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def preproc(images, size): 
    results = np.ndarray([images.shape[0]]+size, np.uint8)
    for idx in range(images.shape[0]): 
        distorted     = Preproc.centerCrop(images[idx], size)
        results[idx]  = distorted
    
    return results

def allData(preprocSize=[28, 28, 3]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    return preproc(data, preprocSize), labels, invertedIdx


def generators(BatchSize, preprocSize=[28, 28, 3]):
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
            distorted     = Preproc.randomFlipH(images[idx])
            #distorted     = Preproc.randomShift(distorted, rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            # distorted     = Preproc.randomRotate(images[idx], rng=30)
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

def genData(): 
    batchTrain, batchTest = generators(BatchSize=10000, preprocSize=[28, 28, 3])
    testData, testLabels = next(batchTest)
    testData = np.reshape(testData, [-1])
    with open('CIFAR10_TestData.dat', 'w') as fout: 
        for idx in range(testData.shape[0]): 
            fout.write(str(testData[idx]) + " ")
    with open('CIFAR10_TestLabel.dat', 'w') as fout: 
        for idx in range(testLabels.shape[0]): 
            fout.write(str(testLabels[idx]) + " ")

HParamCIFAR10 = {'NumGPU': 1, 
                 'BatchSize': 200, 
                 'LearningRate': 1e-4, 
                 'MinLearningRate': 1e-4, 
                 'WeightDecay': 0e-4,
                 'ValidateAfter': 300,
                 'LRDecayAfter': 9000,
                 'LRDecayRate': 0.1,
                 'TestSteps': 50, 
                 'TotalSteps': 18000}

HParamCIFAR10_Quant = {'NumGPU': 1, 
                        'BatchSize': 200, 
                        'LearningRate': 1e-4, 
                        'MinLearningRate': 1e-6, 
                        'WeightDecay': 0e-4,
                        'ValidateAfter': 300,
                        'LRDecayAfter': 3000,
                        'LRDecayRate': 0.1,
                        'TestSteps': 50, 
                        'TotalSteps': 6000}

HParamCIFAR10_Approx = {'NumGPU': 1, 
                        'BatchSize': 200, 
                        'LearningRate': 1e-4, 
                        'MinLearningRate': 1e-6, 
                        'WeightDecay': 0e-4,
                        'ValidateAfter': 300,
                        'LRDecayAfter': 3000,
                        'LRDecayRate': 0.1,
                        'TestSteps': 50, 
                        'TotalSteps': 6000}

# HParamCIFAR10 = {'NumGPU': 4, 
#                  'BatchSize': 1000, 
#                  'LearningRate': 1e-3, 
#                  'MinLearningRate': 1e-4, 
#                  'WeightDecay': 1e-4,
#                  'ValidateAfter': 60,
#                  'LRDecayAfter': 3000,
#                  'LRDecayRate': 0.1,
#                  'TestSteps': 10, 
#                  'TotalSteps': 9000}

# HParamCIFAR10_Quant = {'NumGPU': 4, 
#                         'BatchSize': 1000, 
#                         'LearningRate': 1e-4, 
#                         'MinLearningRate': 1e-6, 
#                         'WeightDecay': 1e-4,
#                         'ValidateAfter': 60,
#                         'LRDecayAfter': 1000,
#                         'LRDecayRate': 0.1,
#                         'TestSteps': 10, 
#                         'TotalSteps': 3000}

# HParamCIFAR10_Approx = {'NumGPU': 4, 
#                         'BatchSize': 1000, 
#                         'LearningRate': 1e-4, 
#                         'MinLearningRate': 1e-6, 
#                         'WeightDecay': 1e-4,
#                         'ValidateAfter': 60,
#                         'LRDecayAfter': 1000,
#                         'LRDecayRate': 0.1,
#                         'TestSteps': 10, 
#                         'TotalSteps': 3000}
        
if __name__ == '__main__':
    # net = Net.Net4Classify(inputShape=[28, 28, 3], numClasses=10, \
    #                        body=Net.LeNetBody, HParam=HParamCIFAR10, name='Net4Classify')
    # batchTrain, batchTest = generators(BatchSize=HParamCIFAR10['BatchSize'], preprocSize=[28, 28, 3])
    # net.train(batchTrain, batchTest, pathSave='./ClassifyCIFAR10/netcifar10.ckpt')
    
    # net = Net.Net4Quant(inputShape=[28, 28, 3], numClasses=10, \
    #                     body=Net.LeNetBody_Quant, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Quant')
    # net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Quant.ckpt')
    
    # net = Net.Net4Approx(inputShape=[28, 28, 3], numClasses=10, \
    #                     body=Net.LeNetBody_Quant, pretrained=net, HParam=HParamCIFAR10_Approx, name='Net4Approx')
    # net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Approx.ckpt')
    
    # net = Net.Net4Eval(inputShape=[28, 28, 3], numClasses=10, \
    #                     body=Net.LeNetBody_Eval, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Quant')
    # net.evaluate(batchTest)


    net = Net.Net4Classify(inputShape=[28, 28, 3], numClasses=10, \
                        #    body=Net.VGG16Body, HParam=HParamCIFAR10, name='Net4Classify')
                        #    body=Net.VGG16BNBody, HParam=HParamCIFAR10, name='Net4Classify')
                           body=Net.AlexNetBody, HParam=HParamCIFAR10, name='Net4Classify')
                        #    body=Net.AlexNetBNBody, HParam=HParamCIFAR10, name='Net4Classify')
    batchTrain, batchTest = generators(BatchSize=HParamCIFAR10['BatchSize'], preprocSize=[28, 28, 3])
    net.train(batchTrain, batchTest, pathSave='./ClassifyCIFAR10/netcifar10.ckpt')
    
    # net = Net.Net4Quant(inputShape=[28, 28, 3], numClasses=10, \
    #                     body=Net.AlexNetBody_Quant, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Quant')
    # net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Quant.ckpt')
    
    net = Net.Net4Approx(inputShape=[28, 28, 3], numClasses=10, \
                        body=Net.AlexNetBody_Approx, pretrained=net, HParam=HParamCIFAR10_Approx, name='Net4Approx')
    net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Approx.ckpt')
    
    net = Net.Net4Eval(inputShape=[28, 28, 3], numClasses=10, \
                        body=Net.AlexNetBody_Eval, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Eval')
    net.evaluate(batchTest)
  
# 1. Noise Training, AlexNet BN
    # PORTION = 1.0: float: 89.97; quant: 89.64; approx: 87.87; 
    # PORTION = 0.8: float: 90.71; quant: 90.34; approx: 88.40; 
  
# 1. Noise Training, AlexNet
    # PORTION = 1.0: quant: 88.71; approx: 88.45;    quant: 91.05; approx: 90.01; 
    # PORTION = 0.8: quant 88.47; approx 88.39; 
    # PORTION = 0.6: quant 90.92; approx 89.88; 
    # PORTION = 0.4: quant ; approx ; 
    # PORTION = 0.2: quant 90.61; approx 89.74; 
    # PORTION = 0.0: quant 90.65; approx 89.72; 


# Corrected
# 0.0: 0.8848 -> 0.8756
# 0.2: 0.8814 -> 0.8797
# 0.4: 0.8865 -> 0.8834
# 0.6:  
# 0.8: 0.8847 -> 0.8839
# 1.0: 0.8926 -> 0.8875