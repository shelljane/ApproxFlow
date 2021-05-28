import tensorflow as tf
import Layer

import numpy as np

from Protocol import Net

from tensorflow.python.ops.array_ops import fake_quant_with_min_max_vars
from tensorflow.python.framework import graph_util

FAKEBITS = Layer.FAKEBITS

PORTION = 1.0
FROM = -1.0
TO = 1.0

HParamDefault = {'NumGPU': 1, 
                 'BatchSize': 50, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 1e-5, 
                 'WeightDecay': 1e-5,
                 'ValidateAfter': 1000,
                 'LRDecayAfter': 10000,
                 'LRDecayRate': 0.1,
                 'TestSteps': 200,
                 'TotalSteps': 30000}

class Net4Classify(Net):
    
    def __init__(self, inputShape, numClasses, body, HParam=HParamDefault, name='Net4Classify'): 
        
        Net.__init__(self, HParam, name)
        
        with self._graph.as_default(), tf.device('/cpu:0'), tf.variable_scope(self._name, reuse=tf.AUTO_REUSE): 
            # Inputs
            self._images = tf.placeholder(dtype=tf.float32, shape=[self._HParam['BatchSize']]+inputShape, name='images')
            self._labels = tf.placeholder(dtype=tf.int64, shape=[self._HParam['BatchSize']], name='labels')
            self._numClasses = numClasses
            self._body   = body
            self._optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-8, use_locking=True)
            
            # Network
            if self._HParam['NumGPU'] > 0: 
                self._imagesGroup = tf.split(self._images, self._HParam['NumGPU'], axis=0)
                self._labelsGroup = tf.split(self._labels, self._HParam['NumGPU'], axis=0)
                
                self._gpuBodies = []
                self._gpuInferences = []
                self._gpuAccuracies = []
                self._gpuLosses = []
                self._gpuLayers = []
                self._lossesList = []
                for idx in range(self._HParam['NumGPU']): 
                    with tf.device('/gpu:%d'%idx): 
                        with tf.name_scope('GPU_%d'%idx): 
                            body, layers = self.body(self._imagesGroup[idx]) 
                            self._gpuLayers.append(layers)
                            self._gpuBodies.append(body)
                            self._gpuInferences.append(self.inference(self._gpuBodies[idx])) 
                            self._gpuAccuracies.append(tf.reduce_mean(tf.cast(tf.equal(self._gpuInferences[idx], self._labelsGroup[idx]), tf.float32))) 
                            self._gpuLosses.append(self.getLoss(layers)) 
                            self._gpuLosses[idx] += self.lossFunc(self._gpuBodies[idx], self._labelsGroup[idx])
                
                self._layers = self._gpuLayers[0]
                self._postInit()
                
                for idx in range(self._HParam['NumGPU']): 
                    with tf.device('/gpu:%d'%idx): 
                        with tf.name_scope('GPU_%d'%idx): 
                            self._lossesList.append(self._optimizer.compute_gradients(self._gpuLosses[idx], gate_gradients=0)) 
                
                self._body = tf.concat(self._gpuBodies, axis=0)
                self._inference = tf.concat(self._gpuInferences, axis=0)
                self._body = tf.concat(self._gpuBodies, axis=0)
                self._inference = tf.concat(self._gpuInferences, axis=0)
                self._loss = tf.reduce_mean(tf.concat([tf.expand_dims(elem, axis=0) for elem in self._gpuLosses], axis=0), axis=0)
                self._accuracy = tf.reduce_mean(tf.concat([tf.expand_dims(elem, axis=0) for elem in self._gpuAccuracies], axis=0), axis=0)
                self._updateOps = [] 
                for idx in range(len(self._gpuLayers)): 
                    self._updateOps.extend(self.getUpdateOps(self._gpuLayers[idx]))
                
                applyList = []
                for idx in range(len(self._lossesList[0])): 
                    grads = []
                    for jdx in range(len(self._lossesList)): 
                        grads.append(tf.expand_dims(self._lossesList[jdx][idx][0], axis=0))
                    applyList.append((tf.reduce_mean(tf.concat(grads, axis=0), axis=0), self._lossesList[0][idx][1]))
                self._optimizer = self._optimizer.apply_gradients(applyList, global_step=self._step)
            else:
                body, layers = self.body(self._images) 
                self._body = body
                self._inference = self.inference(self._body) 
                self._loss = self.getLoss(layers)
                self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._inference, self._labels), tf.float32))
                self._layers = layers
                self._updateOps = self.getUpdateOps(layers) 
                
                self._postInit()
                
                applyList = self._optimizer.compute_gradients(self._loss, gate_gradients=0)
                self._optimizer = self._optimizer.apply_gradients(applyList, global_step=self._step)
                    
            # Saver
            self._saver = tf.train.Saver(max_to_keep=5)
            
            # Network Graph
#            self._writer = tf.summary.FileWriter("./Tensorboard", self._sess.graph)
    
    def _postInit(self): 
        pass
        # tf.contrib.quantize.create_training_graph(input_graph=self._graph, quant_delay=0)
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    
    def train(self, genTrain, genTest, pathLoad=None, pathSave=None): 
        
        with self._graph.as_default(): 
            
            # Initialize all
            self._sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            
            if pathLoad is not None:
                self.load(pathLoad)
           
            maxAccu = self.evaluate(genTest)
            self._postTrain('./NoTrainWeights')
            
#            self._writer.close()
            
            self._sess.run([self._phaseTrain])
            
            for _ in range(self._HParam['TotalSteps']): 
                
                data, label = next(genTrain)
                
                loss, accu, step, _ = self._sess.run([self._loss, self._accuracy, self._step, self._optimizer], \
                                                     feed_dict={self._images: data, self._labels: label})
                self._sess.run(self._updateOps)
                print('\rStep: ', step, '; Loss: %.3f'% loss, '; Accuracy: %.3f'% accu, end='')
                
                if step % self._HParam['ValidateAfter'] == 0: 
                    print('\n')
                    accu = self.evaluate(genTest)
                    if pathSave is not None and accu >= maxAccu: 
                        maxAccu = accu
                        self.save(pathSave)
                        self._postTrain()
                    self._sess.run([self._phaseTrain])
                
    def _postTrain(self, path='./QuantWeights'): 
        pass
    
    def evaluate(self, genTest, path=None): 
        
        if path is not None:
            self.load(path)
        
        totalLoss = 0.0
        totalAccu = 0.0
        self._sess.run([self._phaseTest])  
        for idx in range(self._HParam['TestSteps']): 
            data, label = next(genTest)
            loss, accu = self._sess.run([self._loss, self._accuracy], \
                                        feed_dict={self._images: data, \
                                                   self._labels: label})
            totalLoss += loss
            totalAccu += accu
            print('\rTest Step: ', idx, '; Loss: %.3f'% loss, '; Accuracy: %.3f'% accu, end='')
        totalLoss /= self._HParam['TestSteps']
        totalAccu /= self._HParam['TestSteps']
        print('\nTest: Loss: ', totalLoss, '; Accuracy: ', totalAccu, '\n')
        
        return totalAccu

    def body(self, images):
        
        # Body
        net, layers = self._body(self, images)
        
        logits = Layer.FullyConnected(net, outputSize=self._numClasses, \
                                      weightInit=Layer.XavierInit, wd=self._HParam['WeightDecay'], \
                                      biasInit=Layer.ConstInit(0.0), \
                                      activation=Layer.Linear, \
                                    #   bn=True, step=self._step, ifTest=self._ifTest, epsilon=1e-8, \
                                      name='FC_Logits', dtype=tf.float32)
        layers.append(logits)
        
        
        return logits.output, layers
        
    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')
    
    def lossFunc(self, logits, labels, name='cross_entropy'):
        net = Layer.CrossEntropy(logits, labels, name=name)
        return net.output
                
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
        # print(self._sess.graph_def)
        # constantGraph = graph_util.convert_variables_to_constants(self._sess, self._sess.graph_def, ['Net4Classify_1/GPU_0/FC_Logits/FinalOutput'])
        # with tf.gfile.FastGFile("/".join(path.split("/")[:-1]) + "/saved_model.pb", "wb") as fout: 
        #     fout.write(constantGraph.SerializeToString())
    
    def load(self, path):
        self._saver.restore(self._sess, path)


class Net4Quant(Net4Classify):
    
    def __init__(self, inputShape, numClasses, body, pretrained, HParam=HParamDefault, name='Net4Quant'): 
        
        self._pretrained = pretrained
        self._preInit()
        
        Net4Classify.__init__(self, inputShape, numClasses, body, HParam, name)
    
    def _preInit(self): 
        
        # tf.contrib.quantize.create_training_graph(input_graph=self._pretrained._graph, quant_delay=0)
        self._pretrained._quantLayers = {}
        self._pretrained._haveWeights = {}
        self._pretrained._haveBN      = {}
        self._pretrained._layerInfos  = []
        self._pretrained._layerNames  = []
        self._pretrained._layerTypes  = []
        for idx in range(len(self._pretrained._layers)): 
            print('Analyzing layer: ', self._pretrained._layers[idx]._name)
            name = self._pretrained._layers[idx]._name
            layertype = self._pretrained._layers[idx]._type
            self._pretrained._layerNames.append(name)
            self._pretrained._quantLayers[name] = {}
            for varName in self._pretrained._layers[idx]._variables.keys(): 
                if varName == 'Weights': 
                    self._pretrained._haveWeights[name] = True
                    weights = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['Weights'] = weights
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'Bias': 
                    self._pretrained._haveWeights[name] = True
                    bias = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['Bias'] = bias
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_Offset': 
                    self._pretrained._haveBN[name] = True
                    offset = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_Offset'] = offset
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_Scale': 
                    self._pretrained._haveBN[name] = True
                    scale = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_Scale'] = scale
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_MovMean': 
                    self._pretrained._haveBN[name] = True
                    movmean = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_MovMean'] = movmean
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_MovVar': 
                    self._pretrained._haveBN[name] = True
                    movvar = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_MovVar'] = movvar
                    print(self._pretrained._layers[idx]._variables[varName])
            act_min = self._pretrained._layers[idx].outMin
            self._pretrained._quantLayers[name]['Act_Min'] = act_min
            print(act_min)
            act_max = self._pretrained._layers[idx].outMax
            self._pretrained._quantLayers[name]['Act_Max'] = act_max
            print(act_max)
            if layertype.find('Conv') >= 0: 
                info = [self._pretrained._layers[idx]._strideConv[1]]
                if self._pretrained._layers[idx]._pool: 
                    layertype += 'Pooling'
                    info.extend([self._pretrained._layers[idx]._sizePooling[1], self._pretrained._layers[idx]._stridePooling[1]])
                self._pretrained._layerInfos.append(info)
            elif layertype.find('Pooling') >= 0: 
                info = [self._pretrained._layers[idx]._sizePooling[1], self._pretrained._layers[idx]._stridePooling[1]]
                self._pretrained._layerInfos.append(info)
            else: 
                info = []
                self._pretrained._layerInfos.append(info)
            self._pretrained._layerTypes.append(layertype)
            # print(self._pretrained._layers[idx]._variables)
            
        list(map(lambda l: print(l) or list(map(lambda x: print("\t", x, ":", self._pretrained._quantLayers[l][x]), self._pretrained._quantLayers[l].keys())), self._pretrained._layerNames))
        
        self._preWeights = {}
        self._preBias    = {}
        for idx in range(len(self._pretrained._layerNames)): 
            name = self._pretrained._layerNames[idx]
            layertype = self._pretrained._layerTypes[idx]
            print("Copying Layer:", name)
            if name in self._pretrained._haveWeights: 
                weights = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Weights'])
                bias    = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Bias']) if 'Bias' in self._pretrained._quantLayers[name] else 0.0
                act_min = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Act_Min'])
                act_max = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Act_Max'])
                if name in self._pretrained._haveBN: 
                    offset  = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_Offset']) if 'BN_Offset' in self._pretrained._quantLayers[name] else 0.0
                    scale   = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_Scale']) if 'BN_Scale' in self._pretrained._quantLayers[name] else 1.0
                    movmean = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_MovMean'])
                    movvar  = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_MovVar'])
                    stddev = np.sqrt(movvar + 1e-8)
                    tmp = scale / stddev
                    weights = weights * tmp
                    bias    = offset + tmp * (bias - movmean)
                self._preWeights[name] = weights
                self._preBias[name]    = bias

    def body(self, images):
    
        def _outWrapper(net): 
            # Simulate quantization
            a = net._outMin
            b = net._outMax
            s = (b - a) / 255.0
            output = net.output
            output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
            # Simulate value degrade in approximate computing
            # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
            return output
        
        # Body
        net, layers = self._body(self, images, self._preWeights, self._preBias)
        
        logits = Layer.FullyConnected(_outWrapper(net), outputSize=self._numClasses, \
                                      weightInit=Layer.ConstInit(self._preWeights['FC_Logits']), wd=self._HParam['WeightDecay'], \
                                      biasInit=Layer.ConstInit(self._preBias['FC_Logits']), \
                                      activation=Layer.Linear, \
                                      fakeQuant=True, name='FC_Logits', dtype=tf.float32)
        layers.append(logits)
        
        
        return logits.output, layers
    
    def _postInit(self): 
        
        # tf.contrib.quantize.create_training_graph(input_graph=self._graph, quant_delay=0)
        self._quantLayers = {}
        self._haveWeights = {}
        self._haveBN      = {}
        self._layersTable = {}
        self._layerInfos  = []
        self._layerNames  = []
        self._layerTypes  = []
        for idx in range(len(self._layers)): 
            print('Analyzing layer: ', self._layers[idx]._name)
            name = self._layers[idx]._name
            self._layersTable[name] = self._layers[idx]
            layertype = self._layers[idx]._type
            self._layerNames.append(name)
            self._quantLayers[name] = {}
            for varName in self._layers[idx]._variables.keys(): 
                if varName == 'Weights': 
                    self._haveWeights[name] = True
                    weights = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['Weights'] = weights
                    print(self._layers[idx]._variables[varName])
                elif varName == 'Bias': 
                    self._haveWeights[name] = True
                    bias = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['Bias'] = bias
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_Offset': 
                    self._haveBN[name] = True
                    offset = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_Offset'] = offset
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_Scale': 
                    self._haveBN[name] = True
                    scale = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_Scale'] = scale
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_MovMean': 
                    self._haveBN[name] = True
                    movmean = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_MovMean'] = movmean
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_MovVar': 
                    self._haveBN[name] = True
                    movvar = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_MovVar'] = movvar
                    print(self._layers[idx]._variables[varName])
            act_min = self._layers[idx].outMin
            self._quantLayers[name]['Act_Min'] = act_min
            print(act_min)
            act_max = self._layers[idx].outMax
            self._quantLayers[name]['Act_Max'] = act_max
            print(act_max)
            if layertype.find('Conv') >= 0: 
                info = [self._layers[idx]._strideConv[1]]
                if self._layers[idx]._pool: 
                    layertype += 'Pooling'
                    info.extend([self._layers[idx]._sizePooling[1], self._layers[idx]._stridePooling[1]])
                self._layerInfos.append(info)
            elif layertype.find('Pooling') >= 0: 
                info = [self._layers[idx]._sizePooling[1], self._layers[idx]._stridePooling[1]]
                self._layerInfos.append(info)
            else: 
                info = []
                self._layerInfos.append(info)
            self._layerTypes.append(layertype)
            # print(self._layers[idx]._variables)
            
        list(map(lambda l: print(l) or list(map(lambda x: print("\t", x, ":", self._quantLayers[l][x]), self._quantLayers[l].keys())), self._layerNames))
        
    def _postTrain(self, path='./QuantWeights'):
        
        def quantWeights(weights, layerName): 
            #print('Max: ', weights.max(), ';  Min: ', weights.min())
            #S_weights = (weights.max() - weights.min()) / 255
            #Z_weights = int(np.round((0.0 - weights.min()) / S_weights))
            #Q_weights = np.round((weights - weights.min()) / S_weights).astype(np.int)
            maxabs = np.abs(weights).max()
            # maxabs = self._layersTable[layerName]._weightMax.eval(session=self._sess)
            print('Max: ', weights.max(), ';  Min: ', weights.min(), ';  Abs: ', maxabs)
            S_weights = 2 * maxabs / 255.0
            Z_weights = 128
            Q_weights = np.zeros_like(weights).astype(np.int)
            Q_weights[weights == 0] = Z_weights
            Q_weights[weights > 0] = (Z_weights + np.round((weights) / S_weights).astype(np.int))[weights > 0]
            Q_weights[weights < 0] = (Z_weights + np.round((weights) / S_weights).astype(np.int))[weights < 0]
            # Q_weights = np.round((weights + maxabs) / S_weights).astype(np.int)
            Q_weights[Q_weights > 255] = 255
            Q_weights[Q_weights < 0]   = 0
            print(" -> 0:", np.sum(Q_weights == 0), "127:", np.sum(Q_weights == 127), "128:", np.sum(Q_weights == 128), "129:", np.sum(Q_weights == 129), "255:", np.sum(Q_weights == 255))
            
            return S_weights, Z_weights, Q_weights
        
        def quantBias(biases, S_input, S_weights, layerName): 
            print('Max: ', biases.max(), ';  Min: ', biases.min())
            S_biases = S_input * S_weights
            Z_biases = 0
            Q_biases = np.round(biases / S_biases).astype(np.int)
            
            return S_biases, Z_biases, Q_biases
        
        def quantAct(minAct, maxAct): 
            print('Max: ', maxAct, ';  Min: ', minAct)
            S_acts = (maxAct - minAct) / 255
            Z_acts = int(np.round((0.0 - minAct) / S_acts))
            return S_acts, Z_acts
            #maxabs = max(abs(maxAct), abs(minAct))
            #print('Max: ', maxAct, ';  Min: ', minAct, ';  MaxAbs: ', maxabs)
            #S_acts = 2 * maxabs / 255.0
            #Z_acts = 128
            #return S_acts, Z_acts
        
        self._postWeights = {}
        self._postBias    = {}
        self._postActivationsMax = {}
        self._postActivationsMin = {}
        
        S_input = 1.0 / 255.0
        Z_input = 0
        
        S_last  = S_input
        
        f_names  = open(path + '/' + self._name+'_names.txt', 'w')
        f_config = open(path + '/' + self._name+'_config.txt', 'w')
        f_debug  = open('./Debug.txt', 'w')
        
        for idx in range(len(self._layerNames)): 
            name = self._layerNames[idx]
            layertype = self._layerTypes[idx]
            f_names.write(name + " " + layertype + "\n")
            print("Quantizing Layer:", name)
            if name in self._haveWeights: 
                weights = self._sess.run(self._quantLayers[name]['Weights'])
                bias    = self._sess.run(self._quantLayers[name]['Bias'])
                act_min = self._sess.run(self._quantLayers[name]['Act_Min'])
                act_max = self._sess.run(self._quantLayers[name]['Act_Max'])
                if name in self._haveBN: 
                    offset  = self._sess.run(self._quantLayers[name]['BN_Offset'])
                    scale   = self._sess.run(self._quantLayers[name]['BN_Scale'])
                    movmean = self._sess.run(self._quantLayers[name]['BN_MovMean'])
                    movvar  = self._sess.run(self._quantLayers[name]['BN_MovVar'])
                    assert len(offset.shape) == 1, 'WRONG: offset'
                    assert len(scale.shape) == 1, 'WRONG: scale'
                    assert len(movmean.shape) == 1, 'WRONG: movmean'
                    assert len(movvar.shape) == 1, 'WRONG: movvar'
                    stddev = np.sqrt(movvar + 1e-8)
                    tmp = scale / stddev
                    weights = weights * tmp
                    bias    = offset + tmp * (bias - movmean)
                
                shape_weights = weights.shape
                shape_bias    = bias.shape
                
                for jdx in range(len(shape_weights)): 
                    f_config.write(str(shape_weights[jdx]) + " ")
                for info in self._layerInfos[idx]: 
                    f_config.write(str(info) + " ") 
                f_config.write("\n")
                
                weights = weights.reshape([-1])
                bias    = bias.reshape([-1])
                
                S_weights, Z_weights, Q_weights = quantWeights(weights, name)
                S_biases, Z_biases, Q_biases    = quantBias(bias, S_last, S_weights, name)
                S_acts, Z_acts                  = quantAct(act_min, act_max)
                S_last = S_acts
                
                self._postWeights[name] = S_weights * (Q_weights - Z_weights)
                self._postBias[name] = S_biases * (Q_biases - Z_biases)
                self._postActivationsMin[name] = act_min
                self._postActivationsMax[name] = act_max
                
                print(name, ' weights: ', file = f_debug)
                print(self._postWeights[name], file = f_debug)
                print(name, ' bias: ', file = f_debug)
                print(self._postBias[name], file = f_debug)
                
                with open(path + '/' + name + '_weights.txt', 'w') as fout: 
                    fout.write(str(S_weights) + "\n")
                    fout.write(str(Z_weights) + "\n")
                    for idx in range(Q_weights.shape[0]): 
                        fout.write(str(Q_weights[idx]) + " ")
            
                with open(path + '/' + name + '_biases.txt', 'w') as fout: 
                    fout.write(str(S_biases) + "\n")
                    fout.write(str(Z_biases) + "\n")
                    for idx in range(Q_biases.shape[0]): 
                        fout.write(str(Q_biases[idx]) + " ")
            
                with open(path + '/' + name + '_activations.txt', 'w') as fout: 
                    fout.write(str(S_acts) + "\n")
                    fout.write(str(Z_acts) + "\n")
            else: 
                act_min = self._sess.run(self._quantLayers[name]['Act_Min'])
                act_max = self._sess.run(self._quantLayers[name]['Act_Max'])
                S_acts, Z_acts = quantAct(act_min, act_max)
                S_last = S_acts
                
                self._postActivationsMin[name] = act_min
                self._postActivationsMax[name] = act_max
                
                if layertype.find('Pooling') >= 0: 
                    f_config.write(str(self._layerInfos[idx][0]) + " " + str(self._layerInfos[idx][1])) 
                f_config.write("\n")
                
                with open(path + '/' + name + '_activations.txt', 'w') as fout: 
                    fout.write(str(S_acts) + "\n")
                    fout.write(str(Z_acts) + "\n")
        f_names.close()
        f_config.close()
        f_debug.close()


class Net4Approx(Net4Classify):
    
    def __init__(self, inputShape, numClasses, body, pretrained, HParam=HParamDefault, name='Net4Approx'): 
        
        self._pretrained = pretrained
        self._preInit()
        
        Net4Classify.__init__(self, inputShape, numClasses, body, HParam, name)
    
    def _preInit(self): 
        
        # tf.contrib.quantize.create_training_graph(input_graph=self._pretrained._graph, quant_delay=0)
        self._pretrained._quantLayers = {}
        self._pretrained._haveWeights = {}
        self._pretrained._haveBN      = {}
        self._pretrained._layerInfos  = []
        self._pretrained._layerNames  = []
        self._pretrained._layerTypes  = []
        for idx in range(len(self._pretrained._layers)): 
            print('Analyzing layer: ', self._pretrained._layers[idx]._name)
            name = self._pretrained._layers[idx]._name
            layertype = self._pretrained._layers[idx]._type
            self._pretrained._layerNames.append(name)
            self._pretrained._quantLayers[name] = {}
            for varName in self._pretrained._layers[idx]._variables.keys(): 
                if varName == 'Weights': 
                    self._pretrained._haveWeights[name] = True
                    weights = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['Weights'] = weights
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'Bias': 
                    self._pretrained._haveWeights[name] = True
                    bias = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['Bias'] = bias
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_Offset': 
                    self._pretrained._haveBN[name] = True
                    offset = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_Offset'] = offset
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_Scale': 
                    self._pretrained._haveBN[name] = True
                    scale = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_Scale'] = scale
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_MovMean': 
                    self._pretrained._haveBN[name] = True
                    movmean = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_MovMean'] = movmean
                    print(self._pretrained._layers[idx]._variables[varName])
                elif varName == 'BN_MovVar': 
                    self._pretrained._haveBN[name] = True
                    movvar = self._pretrained._layers[idx]._variables[varName]
                    self._pretrained._quantLayers[name]['BN_MovVar'] = movvar
                    print(self._pretrained._layers[idx]._variables[varName])
            act_min = self._pretrained._layers[idx].outMin
            self._pretrained._quantLayers[name]['Act_Min'] = act_min
            print(act_min)
            act_max = self._pretrained._layers[idx].outMax
            self._pretrained._quantLayers[name]['Act_Max'] = act_max
            print(act_max)
            if layertype.find('Conv') >= 0: 
                info = [self._pretrained._layers[idx]._strideConv[1]]
                if self._pretrained._layers[idx]._pool: 
                    layertype += 'Pooling'
                    info.extend([self._pretrained._layers[idx]._sizePooling[1], self._pretrained._layers[idx]._stridePooling[1]])
                self._pretrained._layerInfos.append(info)
            elif layertype.find('Pooling') >= 0: 
                info = [self._pretrained._layers[idx]._sizePooling[1], self._pretrained._layers[idx]._stridePooling[1]]
                self._pretrained._layerInfos.append(info)
            else: 
                info = []
                self._pretrained._layerInfos.append(info)
            self._pretrained._layerTypes.append(layertype)
            # print(self._pretrained._layers[idx]._variables)
            
        list(map(lambda l: print(l) or list(map(lambda x: print("\t", x, ":", self._pretrained._quantLayers[l][x]), self._pretrained._quantLayers[l].keys())), self._pretrained._layerNames))
        
        self._preWeights = {}
        self._preBias    = {}
        for idx in range(len(self._pretrained._layerNames)): 
            name = self._pretrained._layerNames[idx]
            layertype = self._pretrained._layerTypes[idx]
            print("Copying Layer:", name)
            if name in self._pretrained._haveWeights: 
                weights = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Weights'])
                bias    = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Bias']) if 'Bias' in self._pretrained._quantLayers[name] else 0.0
                act_min = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Act_Min'])
                act_max = self._pretrained._sess.run(self._pretrained._quantLayers[name]['Act_Max'])
                if name in self._pretrained._haveBN: 
                    offset  = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_Offset']) if 'BN_Offset' in self._pretrained._quantLayers[name] else 0.0
                    scale   = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_Scale']) if 'BN_Scale' in self._pretrained._quantLayers[name] else 1.0
                    movmean = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_MovMean'])
                    movvar  = self._pretrained._sess.run(self._pretrained._quantLayers[name]['BN_MovVar'])
                    stddev = np.sqrt(movvar + 1e-8)
                    tmp = scale / stddev
                    weights = weights * tmp
                    bias    = offset + tmp * (bias - movmean)
                self._preWeights[name] = weights
                self._preBias[name]    = bias

    def body(self, images):
    
        def _outWrapper(net): 
            # Simulate quantization
            a = net._outMin
            b = net._outMax
            s = (b - a) / 255.0
            output = net.output
            output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
            # Simulate value degrade in approximate computing
            # output += PORTION * (output - tf.reduce_min(output)) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
            output += PORTION * tf.abs(output) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
            return output
        
        # Body
        net, layers = self._body(self, images, self._preWeights, self._preBias)
        
        logits = Layer.FullyConnected(_outWrapper(net), outputSize=self._numClasses, \
                                      weightInit=Layer.ConstInit(self._preWeights['FC_Logits']), wd=self._HParam['WeightDecay'], \
                                      biasInit=Layer.ConstInit(self._preBias['FC_Logits']), \
                                      activation=Layer.Linear, \
                                      fakeQuant=True, name='FC_Logits', dtype=tf.float32)
        layers.append(logits)
        
        
        return logits.output, layers
    
    def _postInit(self): 
        
        # tf.contrib.quantize.create_training_graph(input_graph=self._graph, quant_delay=0)
        self._quantLayers = {}
        self._layersTable = {}
        self._haveWeights = {}
        self._haveBN      = {}
        self._layerInfos  = []
        self._layerNames  = []
        self._layerTypes  = []
        for idx in range(len(self._layers)): 
            print('Analyzing layer: ', self._layers[idx]._name)
            name = self._layers[idx]._name
            layertype = self._layers[idx]._type
            self._layersTable[name] = self._layers[idx]
            self._layerNames.append(name)
            self._quantLayers[name] = {}
            for varName in self._layers[idx]._variables.keys(): 
                if varName == 'Weights': 
                    self._haveWeights[name] = True
                    weights = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['Weights'] = weights
                    print(self._layers[idx]._variables[varName])
                elif varName == 'Bias': 
                    self._haveWeights[name] = True
                    bias = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['Bias'] = bias
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_Offset': 
                    self._haveBN[name] = True
                    offset = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_Offset'] = offset
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_Scale': 
                    self._haveBN[name] = True
                    scale = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_Scale'] = scale
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_MovMean': 
                    self._haveBN[name] = True
                    movmean = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_MovMean'] = movmean
                    print(self._layers[idx]._variables[varName])
                elif varName == 'BN_MovVar': 
                    self._haveBN[name] = True
                    movvar = self._layers[idx]._variables[varName]
                    self._quantLayers[name]['BN_MovVar'] = movvar
                    print(self._layers[idx]._variables[varName])
            act_min = self._layers[idx].outMin
            self._quantLayers[name]['Act_Min'] = act_min
            print(act_min)
            act_max = self._layers[idx].outMax
            self._quantLayers[name]['Act_Max'] = act_max
            print(act_max)
            if layertype.find('Conv') >= 0: 
                info = [self._layers[idx]._strideConv[1]]
                if self._layers[idx]._pool: 
                    layertype += 'Pooling'
                    info.extend([self._layers[idx]._sizePooling[1], self._layers[idx]._stridePooling[1]])
                self._layerInfos.append(info)
            elif layertype.find('Pooling') >= 0: 
                info = [self._layers[idx]._sizePooling[1], self._layers[idx]._stridePooling[1]]
                self._layerInfos.append(info)
            else: 
                info = []
                self._layerInfos.append(info)
            self._layerTypes.append(layertype)
            # print(self._layers[idx]._variables)
            
        list(map(lambda l: print(l) or list(map(lambda x: print("\t", x, ":", self._quantLayers[l][x]), self._quantLayers[l].keys())), self._layerNames))
        
    def _postTrain(self, path='./ApproxWeights'):
        
        def quantWeights(weights, layerName): 
            #print('Max: ', weights.max(), ';  Min: ', weights.min())
            #S_weights = (weights.max() - weights.min()) / 255
            #Z_weights = int(np.round((0.0 - weights.min()) / S_weights))
            #Q_weights = np.round((weights - weights.min()) / S_weights).astype(np.int)
            maxabs = np.abs(weights).max()
            # maxabs = self._layersTable[layerName]._weightMax.eval(session=self._sess)
            print('Max: ', weights.max(), ';  Min: ', weights.min(), ';  Abs: ', maxabs)
            S_weights = 2 * maxabs / 255.0
            Z_weights = 128
            Q_weights = np.zeros_like(weights).astype(np.int)
            Q_weights[weights == 0] = Z_weights
            Q_weights[weights > 0] = (Z_weights + np.round((weights) / S_weights).astype(np.int))[weights > 0]
            Q_weights[weights < 0] = (Z_weights + np.round((weights) / S_weights).astype(np.int))[weights < 0]
            # Q_weights = np.round((weights + maxabs) / S_weights).astype(np.int)
            Q_weights[Q_weights > 255] = 255
            Q_weights[Q_weights < 0]   = 0
            print(" -> 0:", np.sum(Q_weights == 0), "127:", np.sum(Q_weights == 127), "128:", np.sum(Q_weights == 128), "129:", np.sum(Q_weights == 129), "255:", np.sum(Q_weights == 255))
            
            
            return S_weights, Z_weights, Q_weights
        
        def quantBias(biases, S_input, S_weights, layerName): 
            print('Max: ', biases.max(), ';  Min: ', biases.min())
            S_biases = S_input * S_weights
            Z_biases = 0
            Q_biases = np.round(biases / S_biases).astype(np.int)
            
            return S_biases, Z_biases, Q_biases
        
        def quantAct(minAct, maxAct): 
            print('Max: ', maxAct, ';  Min: ', minAct)
            S_acts = (maxAct - minAct) / 255
            Z_acts = int(np.round((0.0 - minAct) / S_acts))
            return S_acts, Z_acts
            #maxabs = max(abs(maxAct), abs(minAct))
            #print('Max: ', maxAct, ';  Min: ', minAct, ';  MaxAbs: ', maxabs)
            #S_acts = 2 * maxabs / 255.0
            #Z_acts = 128
            #return S_acts, Z_acts
        
        self._postWeights = {}
        self._postBias    = {}
        self._postActivationsMax = {}
        self._postActivationsMin = {}
        
        S_input = 1.0 / 255.0
        Z_input = 0
        
        S_last  = S_input
        
        f_names  = open('./ApproxWeights/' + self._name+'_names.txt', 'w')
        f_config = open('./ApproxWeights/' + self._name+'_config.txt', 'w')
        f_debug  = open('./Debug.txt', 'w')
        
        for idx in range(len(self._layerNames)): 
            name = self._layerNames[idx]
            layertype = self._layerTypes[idx]
            f_names.write(name + " " + layertype + "\n")
            print("Quantizing Layer:", name)
            if name in self._haveWeights: 
                weights = self._sess.run(self._quantLayers[name]['Weights'])
                bias    = self._sess.run(self._quantLayers[name]['Bias'])
                act_min = self._sess.run(self._quantLayers[name]['Act_Min'])
                act_max = self._sess.run(self._quantLayers[name]['Act_Max'])
                if name in self._haveBN: 
                    offset  = self._sess.run(self._quantLayers[name]['BN_Offset'])
                    scale   = self._sess.run(self._quantLayers[name]['BN_Scale'])
                    movmean = self._sess.run(self._quantLayers[name]['BN_MovMean'])
                    movvar  = self._sess.run(self._quantLayers[name]['BN_MovVar'])
                    assert len(offset.shape) == 1, 'WRONG: offset'
                    assert len(scale.shape) == 1, 'WRONG: scale'
                    assert len(movmean.shape) == 1, 'WRONG: movmean'
                    assert len(movvar.shape) == 1, 'WRONG: movvar'
                    stddev = np.sqrt(movvar + 1e-8)
                    tmp = scale / stddev
                    weights = weights * tmp
                    bias    = offset + tmp * (bias - movmean)
                
                shape_weights = weights.shape
                shape_bias    = bias.shape
                
                for jdx in range(len(shape_weights)): 
                    f_config.write(str(shape_weights[jdx]) + " ")
                for info in self._layerInfos[idx]: 
                    f_config.write(str(info) + " ") 
                f_config.write("\n")
                
                weights = weights.reshape([-1])
                bias    = bias.reshape([-1])
                
                S_weights, Z_weights, Q_weights = quantWeights(weights, name)
                S_biases, Z_biases, Q_biases    = quantBias(bias, S_last, S_weights, name)
                S_acts, Z_acts                  = quantAct(act_min, act_max)
                S_last = S_acts
                
                self._postWeights[name] = S_weights * (Q_weights - Z_weights)
                self._postBias[name] = S_biases * (Q_biases - Z_biases)
                self._postActivationsMin[name] = act_min
                self._postActivationsMax[name] = act_max
                
                print(name, ' weights: ', file = f_debug)
                print(self._postWeights[name], file = f_debug)
                print(name, ' bias: ', file = f_debug)
                print(self._postBias[name], file = f_debug)
                
                with open('./ApproxWeights/' + name + '_weights.txt', 'w') as fout: 
                    fout.write(str(S_weights) + "\n")
                    fout.write(str(Z_weights) + "\n")
                    for idx in range(Q_weights.shape[0]): 
                        fout.write(str(Q_weights[idx]) + " ")
            
                with open('./ApproxWeights/' + name + '_biases.txt', 'w') as fout: 
                    fout.write(str(S_biases) + "\n")
                    fout.write(str(Z_biases) + "\n")
                    for idx in range(Q_biases.shape[0]): 
                        fout.write(str(Q_biases[idx]) + " ")
            
                with open('./ApproxWeights/' + name + '_activations.txt', 'w') as fout: 
                    fout.write(str(S_acts) + "\n")
                    fout.write(str(Z_acts) + "\n")
            else: 
                act_min = self._sess.run(self._quantLayers[name]['Act_Min'])
                act_max = self._sess.run(self._quantLayers[name]['Act_Max'])
                S_acts, Z_acts = quantAct(act_min, act_max)
                S_last = S_acts
                
                self._postActivationsMin[name] = act_min
                self._postActivationsMax[name] = act_max
                
                if layertype.find('Pooling') >= 0: 
                    f_config.write(str(self._layerInfos[idx][0]) + " " + str(self._layerInfos[idx][1])) 
                f_config.write("\n")
                
                with open('./ApproxWeights/' + name + '_activations.txt', 'w') as fout: 
                    fout.write(str(S_acts) + "\n")
                    fout.write(str(Z_acts) + "\n")
        f_names.close()
        f_config.close()
        f_debug.close()

class Net4Eval(Net4Classify):
    
    def __init__(self, inputShape, numClasses, body, pretrained, HParam=HParamDefault, name='Net4Quant'): 
        
        self._pretrained = pretrained
        self._preInit()
        
        Net4Classify.__init__(self, inputShape, numClasses, body, HParam, name)
        
        with self._graph.as_default(): 
            self._sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
    def _preInit(self): 
        
        self._preWeights = self._pretrained._postWeights
        self._preBias    = self._pretrained._postBias
        self._preActivationsMin = self._pretrained._postActivationsMin
        self._preActivationsMax = self._pretrained._postActivationsMax

    def body(self, images):
    
        def _outWrapper(net): 
            # Simulate quantization
            a = net._outMin
            b = net._outMax
            s = (b - a) / 255.0
            output = net.output
            # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
            # Simulate value degrade in approximate computing
            # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
            return output
        
        # Body
        net, layers = self._body(self, images, self._preWeights, self._preBias, self._preActivationsMin, self._preActivationsMax)
        
        logits = Layer.FullyConnected(_outWrapper(net), outputSize=self._numClasses, \
                                      weightInit=Layer.ConstInit(self._preWeights['FC_Logits']), wd=self._HParam['WeightDecay'], \
                                      biasInit=Layer.ConstInit(self._preBias['FC_Logits']), \
                                      activation=Layer.Linear, \
                                      name='FC_Logits', dtype=tf.float32)
        logits.setMinMax(self._preActivationsMin['FC_Logits'], self._preActivationsMax['FC_Logits'])
        layers.append(logits)
        
        
        return logits.output, layers
    
    def saveMiddle(self, image): 
        np.set_printoptions(threshold=np.inf)
        fout = open('Middle.txt', 'w')
        print('Image: ', file=fout)
        print(image, file=fout)
        for layer in self._layers: 
            name = layer._name
            print('Result: ', name, file=fout)
            result = self._sess.run(layer._output, feed_dict={tf._images: image})[0]
            print(result, file=fout)
    
    def _postInit(self): 
        
        pass
        
    def _postTrain(self):
        
        pass

def LeNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=256, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def LeNetBNBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=256, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def LeNetBody_Quant(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=256, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def LeNetBody_Approx(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output += PORTION * (output - tf.reduce_min(output)) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        output += PORTION * tf.abs(output) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=256, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def LeNetBody_Eval(network, images, preWeights, preBias, preMin, preMax): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    net.setMinMax(preMin['Conv1'], preMax['Conv1'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    net.setMinMax(preMin['Conv2'], preMax['Conv2'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    net.setMinMax(preMin['Conv3'], preMax['Conv3'])
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=256, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    net.setMinMax(preMin['FC1'], preMax['FC1'])
    layers.append(net)
    
    return net, layers

def LeNetBigBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def LeNetBigBNBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def LeNetBigBody_Quant(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def LeNetBigBody_Approx(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output += PORTION * (output - tf.reduce_min(output)) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        output += PORTION * tf.abs(output) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def LeNetBigBody_Eval(network, images, preWeights, preBias, preMin, preMax): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    net.setMinMax(preMin['Conv1'], preMax['Conv1'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    net.setMinMax(preMin['Conv2'], preMax['Conv2'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[5, 5], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    net.setMinMax(preMin['Conv3'], preMax['Conv3'])
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    net.setMinMax(preMin['FC1'], preMax['FC1'])
    layers.append(net)
    
    return net, layers

def LargeNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def LargeNetBody_Quant(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def LargeNetBody_Eval(network, images, preWeights, preBias, preMin, preMax): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    net.setMinMax(preMin['Conv1a'], preMax['Conv1a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    net.setMinMax(preMin['Conv1b'], preMax['Conv1b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    net.setMinMax(preMin['Conv2a'], preMax['Conv2a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    net.setMinMax(preMin['Conv2b'], preMax['Conv2b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    net.setMinMax(preMin['Conv3a'], preMax['Conv3a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    net.setMinMax(preMin['Conv3b'], preMax['Conv3b'])
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=512, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    net.setMinMax(preMin['FC1'], preMax['FC1'])
    layers.append(net)
    
    return net, layers


def AlexNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=4096, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(net.output, outputSize=4096, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers


def AlexNetBNBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=4096, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(net.output, outputSize=4096, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def AlexNetBody_Quant(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(_outWrapper(net), outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC2']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC2']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def AlexNetBody_Approx(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= PORTION * (output - tf.reduce_min(output)) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        output += PORTION * tf.abs(output) * tf.random_uniform(minval=FROM, maxval=TO, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(_outWrapper(net), outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC2']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC2']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def AlexNetBody_Eval(network, images, preWeights, preBias, preMin, preMax): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    net.setMinMax(preMin['Conv1a'], preMax['Conv1a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    net.setMinMax(preMin['Conv1b'], preMax['Conv1b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    net.setMinMax(preMin['Conv2a'], preMax['Conv2a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    net.setMinMax(preMin['Conv2b'], preMax['Conv2b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    net.setMinMax(preMin['Conv3a'], preMax['Conv3a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    net.setMinMax(preMin['Conv3b'], preMax['Conv3b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    net.setMinMax(preMin['Conv4a'], preMax['Conv4a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    net.setMinMax(preMin['Conv4b'], preMax['Conv4b'])
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    net.setMinMax(preMin['FC1'], preMax['FC1'])
    layers.append(net)
    net = Layer.FullyConnected(_outWrapper(net), outputSize=4096, weightInit=Layer.ConstInit(preWeights['FC2']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC2']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    net.setMinMax(preMin['FC2'], preMax['FC2'])
    layers.append(net)
    
    return net, layers



def VGG16Body(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        #bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5c', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=1024, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(net.output, outputSize=1024, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                #bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def VGG16BNBody(network, images): 
    layers = []
    standardized = tf.identity(images / 255.0, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        bias=True, biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5c', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=1024, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(net.output, outputSize=1024, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                bias=True, biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers

def VGG16Body_Quant(network, images, preWeights, preBias): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv1b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv2b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv3c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv4c', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv5a', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv5b', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        fakeQuant=True, name='Conv5c', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=1024, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layer.FullyConnected(_outWrapper(net), outputSize=1024, weightInit=Layer.ConstInit(preWeights['FC2']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC2']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                fakeQuant=True, name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net, layers

def VGG16Body_Eval(network, images, preWeights, preBias, preMin, preMax): 
    
    def _outWrapper(net): 
        # Simulate quantization
        a = net._outMin
        b = net._outMax
        s = (b - a) / 255.0
        output = net.output
        # output = fake_quant_with_min_max_vars(net.output, a, b, num_bits=FAKEBITS, narrow_range=False)
        # Simulate value degrade in approximate computing
        # output -= 0.2 * (output - tf.reduce_min(output)) * tf.random_uniform(minval=0.0, maxval=1.0, shape=output.shape)
        return output
    
    layers = []
    standardized = tf.identity(images * (1 / 255.0), name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1a', dtype=tf.float32)
    net.setMinMax(preMin['Conv1a'], preMax['Conv1a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv1b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv1b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv1b', dtype=tf.float32)
    net.setMinMax(preMin['Conv1b'], preMax['Conv1b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2a', dtype=tf.float32)
    net.setMinMax(preMin['Conv2a'], preMax['Conv2a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv2b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv2b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv2b', dtype=tf.float32)
    net.setMinMax(preMin['Conv2b'], preMax['Conv2b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3a', dtype=tf.float32)
    net.setMinMax(preMin['Conv3a'], preMax['Conv3a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3b', dtype=tf.float32)
    net.setMinMax(preMin['Conv3b'], preMax['Conv3b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv3c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv3c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv3c', dtype=tf.float32)
    net.setMinMax(preMin['Conv3c'], preMax['Conv3c'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    net.setMinMax(preMin['Conv4a'], preMax['Conv4a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4b', dtype=tf.float32)
    net.setMinMax(preMin['Conv4b'], preMax['Conv4b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv4c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv4c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv4c', dtype=tf.float32)
    net.setMinMax(preMin['Conv4c'], preMax['Conv4c'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5a']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5a']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5a', dtype=tf.float32)
    net.setMinMax(preMin['Conv5a'], preMax['Conv5a'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5b']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5b']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        # pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5b', dtype=tf.float32)
    net.setMinMax(preMin['Conv5b'], preMax['Conv5b'])
    layers.append(net)
    net = Layer.Conv2D(_outWrapper(net), convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.ConstInit(preWeights['Conv5c']), convPadding='SAME', \
                        biasInit=Layer.ConstInit(preBias['Conv5c']), \
                        # bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], poolType=Layer.MaxPool, poolPadding='SAME', \
                        activation=Layer.ReLU, \
                        name='Conv5c', dtype=tf.float32)
    net.setMinMax(preMin['Conv5c'], preMax['Conv5c'])
    layers.append(net)
    flattened = tf.reshape(_outWrapper(net), [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=1024, weightInit=Layer.ConstInit(preWeights['FC1']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC1']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    net.setMinMax(preMin['FC1'], preMax['FC1'])
    layers.append(net)
    net = Layer.FullyConnected(_outWrapper(net), outputSize=1024, weightInit=Layer.ConstInit(preWeights['FC2']), wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(preBias['FC2']), \
                                # bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC2', dtype=tf.float32)
    net.setMinMax(preMin['FC2'], preMax['FC2'])
    layers.append(net)
    
    return net, layers


# Trash Bin

def SmallNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 127.5 - 1, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv4', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layer.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv5', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-8, \
                        activation=Layer.ReLU, \
                        name='Conv6', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layer.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv7', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Conv8', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layer.FullyConnected(flattened, outputSize=1024, weightInit=Layer.XavierInit, wd=network._HParam['WeightDecay'], \
                                biasInit=Layer.ConstInit(0.0), \
                                bn=True, step=network._step, ifTest=network._ifTest, \
                                activation=Layer.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net.output, layers


def SimpleNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 127.5 - 1, name='images_standardized')
    net = Layer.DepthwiseConv2D(standardized, convChannels=3*16, \
                    convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                    convInit=Layer.XavierInit, convPadding='SAME', \
                    biasInit=Layer.ConstInit(0.0), \
                    bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                    activation=Layer.ReLU, \
                    name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layer.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layer.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.Activation(added, activation=Layer.ReLU, name='ReLU384')
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layer.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.Activation(added, activation=Layer.ReLU, name='ReLU768')
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layer.Activation(added, activation=Layer.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=1024, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layer.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net.output, layers


def ConcatNetBody(network, images): 
    layers = []
    standardized = tf.identity(images / 127.5 - 1, name='images_standardized')
    net = Layer.DepthwiseConv2D(standardized, convChannels=3*16, \
                    convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                    convInit=Layer.XavierInit, convPadding='SAME', \
                    biasInit=Layer.ConstInit(0.0), \
                    bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                    name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layer.Conv2D(net.output, convChannels=48, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage1_Conv_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layer.Conv2D(toconcat.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layer.DepthwiseConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layer.Conv2D(concated, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage2_Conv_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layer.Conv2D(toconcat.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layer.DepthwiseConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layer.Conv2D(concated, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layer.Conv2D(toconcat.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layer.DepthwiseConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layer.Conv2D(concated, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='Stage4_Conv_384a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layer.Conv2D(toconcat.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage4_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layer.DepthwiseConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Stage4_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='Stage4_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    toadd = Layer.Conv2D(concated, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='SepConv768Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(network._numMiddle):
        net = Layer.Activation(conved, Layer.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=768, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layer.Activation(net.output, Layer.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=768, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layer.Activation(net.output, Layer.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=768, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layer.Conv2D(conved, convChannels=1536, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.Activation(conved, Layer.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layer.Conv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layer.Conv2D(toconcat.output, convChannels=1536, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers.append(net)
    net = Layer.DepthwiseConv2D(net.output, convChannels=1536, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.Linear, \
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layer.SepConv2D(added, convChannels=2048, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net.output, layers


def XcepCIFAR(network, images): 
    layers = []
    standardized = tf.identity(images / 127.5 - 1, name='images_standardized')
    net = Layer.Conv2D(standardized, convChannels=32, \
                    convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                    convInit=Layer.XavierInit, convPadding='SAME', \
                    biasInit=Layer.ConstInit(0.0), \
                    bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                    activation=Layer.ReLU, \
                    name='ConvEntry32_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvEntry64_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layer.Conv2D(net.output, convChannels=128, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layer.Conv2D(added, convChannels=256, \
                        convKernel=[1, 1], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layer.Activation(added, Layer.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layer.SepConv2D(acted.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layer.Conv2D(added, convChannels=728, \
                        convKernel=[1, 1], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layer.Activation(added, Layer.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layer.SepConv2D(acted.output, convChannels=728, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=728, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(network._numMiddle): 
        net = Layer.Activation(conved, Layer.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=728, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layer.Activation(net.output, Layer.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=728, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layer.Activation(net.output, Layer.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layer.SepConv2D(net.output, convChannels=728, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                            convInit=Layer.XavierInit, convPadding='SAME', \
                            biasInit=Layer.ConstInit(0.0), \
                            bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                            name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layer.Conv2D(conved, convChannels=1024, \
                        convKernel=[1, 1], convStride=[2, 2], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layer.Activation(conved, Layer.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=728, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=1024, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layer.MaxPool, poolPadding='SAME', \
                        name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layer.SepConv2D(added, convChannels=1536, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.SepConv2D(net.output, convChannels=2048, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=network._HParam['WeightDecay'], \
                        convInit=Layer.XavierInit, convPadding='SAME', \
                        biasInit=Layer.ConstInit(0.0), \
                        bn=True, step=network._step, ifTest=network._ifTest, epsilon=1e-5, \
                        activation=Layer.ReLU, \
                        name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layer.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net.output, layers


