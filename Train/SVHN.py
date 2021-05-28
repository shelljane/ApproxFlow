import scipy.io as sio
import numpy as np
import random
import h5py

import tensorflow as tf

import Preproc
import Layer
import Net

def loadHDF5():
    with h5py.File('SVHN.h5', 'r') as f:
        dataTrain   = np.array(f['Train']['images']) * 255
        labelsTrain = np.array(f['Train']['labels'])
        dataTest    = np.array(f['Test']['images']) * 255
        labelsTest  = np.array(f['Test']['labels'])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def dumpHDF5(one_hot=False):
    train = sio.loadmat('train_32x32.mat')
    test = sio.loadmat('test_32x32.mat')

    train_data = train['X']
    train_label = train['y']
    test_data = test['X']
    test_label = test['y']

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 2)

    test_data = test_data / 255.
    train_data = train_data / 255.

    for i in range(train_label.shape[0]):
        if train_label[i][0] == 10:
            train_label[i][0] = 0

    for i in range(test_label.shape[0]):
        if test_label[i][0] == 10:
            test_label[i][0] = 0

    if one_hot:
        train_label = (np.arange(num_labels) == train_label[:, ]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:, ]).astype(np.float32)

    with h5py.File('SVHN.h5', 'w') as f:
        train = f.create_group("Train")
        test = f.create_group("Test")
        train['images'] = train_data
        train['labels'] = train_label.reshape([-1])
        test['images'] = test_data
        test['labels'] = test_label.reshape([-1])

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
    with open('SVHN_TestData.dat', 'w') as fout: 
        for idx in range(testData.shape[0]): 
            fout.write(str(testData[idx]) + " ")
    with open('SVHN_TestLabel.dat', 'w') as fout: 
        for idx in range(testLabels.shape[0]): 
            fout.write(str(testLabels[idx]) + " ")

HParamCIFAR10 = {'NumGPU': 1, 
                 'BatchSize': 200, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 1e-5, 
                 'WeightDecay': 1e-5,
                 'ValidateAfter': 300,
                 'LRDecayAfter': 6000,
                 'LRDecayRate': 0.1,
                 'TestSteps': 50, 
                 'TotalSteps': 6000}

HParamCIFAR10_Quant = {'NumGPU': 1, 
                        'BatchSize': 200, 
                        'LearningRate': 1e-4, 
                        'MinLearningRate': 1e-6, 
                        'WeightDecay': 1e-5,
                        'ValidateAfter': 300,
                        'LRDecayAfter': 3000,
                        'LRDecayRate': 0.1,
                        'TestSteps': 50, 
                        'TotalSteps': 6000}

HParamCIFAR10_Approx = {'NumGPU': 1, 
                        'BatchSize': 200, 
                        'LearningRate': 1e-4, 
                        'MinLearningRate': 1e-6, 
                        'WeightDecay': 1e-5,
                        'ValidateAfter': 300,
                        'LRDecayAfter': 3000,
                        'LRDecayRate': 0.1,
                        'TestSteps': 50, 
                        'TotalSteps': 6000}
        
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
                           body=Net.LeNetBigBody, HParam=HParamCIFAR10, name='Net4Classify')
    batchTrain, batchTest = generators(BatchSize=HParamCIFAR10['BatchSize'], preprocSize=[28, 28, 3])
    net.train(batchTrain, batchTest, pathSave='./ClassifyCIFAR10/netcifar10.ckpt')
    
    # net = Net.Net4Quant(inputShape=[28, 28, 3], numClasses=10, \
    #                     body=Net.LeNetBigBody_Quant, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Quant')
    # net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Quant.ckpt')
    
    net = Net.Net4Approx(inputShape=[28, 28, 3], numClasses=10, \
                        body=Net.LeNetBigBody_Approx, pretrained=net, HParam=HParamCIFAR10_Approx, name='Net4Approx')
    net.train(batchTrain, batchTest, pathSave='./ClassifyMNIST/netCIFAR10_Approx.ckpt')
    
    net = Net.Net4Eval(inputShape=[28, 28, 3], numClasses=10, \
                        body=Net.LeNetBigBody_Eval, pretrained=net, HParam=HParamCIFAR10_Quant, name='Net4Quant')
    net.evaluate(batchTest)
 
    # dumpHDF5()
    # train_data, train_label, test_data, test_label = loadHDF5()
    # print(train_data.dtype)
    # print(train_data.shape)
    # print(train_label.dtype)
    # print(train_label.shape)
    # print(test_data.dtype)
    # print(test_data.shape)
    # print(test_label.dtype)
    # print(test_label.shape)
