import functools
import tensorflow as tf

from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops.array_ops import fake_quant_with_min_max_vars

from Protocol import Layer

FAKEBITS = 8
# FAKEBITS = 7

# def assign_moving_average(movValue, rawValue, movingRate): 
#     return tf.assign(movValue, movingRate * movValue + (1.0 - movingRate) * rawValue)

# Initializers
XavierInit = tf.contrib.layers.xavier_initializer()
Norm01Init = tf.truncated_normal_initializer(0.0, stddev=0.1)
def NormalInit(stddev, dtype=tf.float32):
    return tf.truncated_normal_initializer(0.0, stddev=stddev, dtype=dtype)
def ConstInit(const, dtype=tf.float32):
    return tf.constant_initializer(const, dtype=dtype)

# Activations
Linear  = tf.identity
Sigmoid = tf.nn.sigmoid
Tanh    = tf.nn.tanh
ReLU    = tf.nn.relu
ELU     = tf.nn.elu
Softmax = tf.nn.softmax
def LeakyReLU(alpha=0.2):
    return functools.partial(tf.nn.leaky_relu, alpha=alpha)

#Poolings
AvgPool        = tf.nn.avg_pool
MaxPool        = tf.nn.max_pool

# Variable
def cpu_variable(name, shapeParams, initializer=ConstInit(0.0), trainable=True, dtype=tf.float32): 
    with tf.device('/cpu:0'): 
        var = tf.get_variable(name, shapeParams, initializer=initializer, dtype=dtype, trainable=trainable)
    return var
    
# Convolution

class Conv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, scale=True, offset=True, step=None, ifTest=None, movingRate=0.9, epsilon=1e-8, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 fakeQuant=False, name=None, dtype=tf.float32): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
                
        Layer.__init__(self)
        
        self._type = 'Conv2D'
        self._name = name
        self._fakeQuant = fakeQuant
        
        with tf.variable_scope(self._name) as scope: 
            self._sizeKernel      = convKernel + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv      = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            self._weights = cpu_variable(scope.name+'_weights', self._sizeKernel, \
                                         initializer=convInit, dtype=dtype)
            if fakeQuant: 
                a = tf.reduce_min(self._weights)
                b = tf.reduce_max(self._weights)
                a = -tf.reduce_max(tf.abs(self._weights))
                b = tf.reduce_max(tf.abs(self._weights))
                self._weightMin = a; 
                self._weightMax = b; 
                self._weights = fake_quant_with_min_max_vars(self._weights, a, b, num_bits=FAKEBITS, narrow_range=False)
            conv = tf.nn.conv2d(feature, self._weights, self._strideConv, padding=self._typeConvPadding, \
                                name=scope.name+'_conv2d')
            self._variables['Weights'] = self._weights
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), convWD, name=scope.name+'l2_wd')
                # decay = tf.multiply(tf.reduce_sum(tf.abs(self._weights)), convWD, name=scope.name+'l1_wd')
                self._losses['L2Decay_Weights'] = decay 
        
            if bias: 
                self._bias = cpu_variable(scope.name+'_bias', [convChannels], \
                                          initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                # if fakeQuant: 
                #     a = tf.reduce_min(self._bias)
                #     b = tf.reduce_max(self._bias)
                #     a = -tf.reduce_max(tf.abs(self._bias))
                #     b = tf.reduce_max(tf.abs(self._bias))
                #     self._biasMin = a; 
                #     self._biasMax = b; 
                #     self._bias = fake_quant_with_min_max_vars(self._bias, a, b, num_bits=FAKEBITS, narrow_range=False)
                self._variables['Bias'] = self._bias
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._bias), convWD, name=scope.name+'l2_bd')
                    # decay = tf.multiply(tf.reduce_sum(tf.abs(self._bias)), convWD, name=scope.name+'l1_bd')
                    self._losses['L2Decay_Bias'] = decay 
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                if offset: 
                    self._offset  = cpu_variable(scope.name+'_offset', \
                                                 shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                else: 
                    self._offset  = cpu_variable(scope.name+'_offset', \
                                                 shapeParams, initializer=ConstInit(0.0), trainable=False, dtype=dtype)
                if scale: 
                    self._scale   = cpu_variable(scope.name+'_scale', \
                                                 shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                else: 
                    self._scale   = cpu_variable(scope.name+'_scale', \
                                                 shapeParams, initializer=ConstInit(1.0), trainable=False, dtype=dtype)
                self._movMean = cpu_variable(scope.name+'_movMean', \
                                             shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = cpu_variable(scope.name+'_movVar', \
                                             shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables['BN_Offset'] = self._offset
                self._variables['BN_Scale'] = self._scale
                self._variables['BN_MovMean'] = self._movMean
                self._variables['BN_MovVar'] = self._movVar
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._scale)+tf.nn.l2_loss(self._offset), convWD, name=scope.name+'l2_bnd')
                    # decay = tf.multiply(tf.reduce_sum(tf.abs(self._scale))+tf.reduce_sum(tf.abs(self._offset)), convWD, name=scope.name+'l1_bnd')
                    self._losses['L2Decay_BN'] = decay 
                
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                                  assign_moving_average(self._movVar, var, movingRate)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                 self._offset, self._scale, self._epsilon, \
                                                 name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            
            self._pool = pool
            if pool:
                self._sizePooling     = [1] + poolSize + [1]
                self._stridePooling   = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, \
                                  padding=self._typePoolPadding, name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
                
        self._output = self._outWrapper(pooled)
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
            
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeKernel) + '; ' + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Activation: ' + activation + ';' + 'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Stride: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + ']')

class SepConv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, movingRate=0.9, epsilon=1e-8, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 name=None, dtype=tf.float32): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        
        self._type = 'SepConv2D'
        self._name = name
        
        with tf.variable_scope(self._name) as scope: 
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3], 1]
            self._sizePointKernel = [1, 1] + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv      = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            
            self._weightsDepth = cpu_variable(scope.name+'_weightsDepth', \
                                              self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            self._weightsPoint = cpu_variable(scope.name+'_weightsPoint', \
                                              self._sizePointKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.separable_conv2d(feature, self._weightsDepth, self._weightsPoint, \
                                          strides=self._strideConv, padding=self._typeConvPadding, \
                                          name=scope.name+'_sep_conv')
            self._variables['WeightsDepth'] = self._weightsDepth
            self._variables['WeightsPoint'] = self._weightsPoint
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), convWD, name=scope.name+'l2_wd_depth')
                self._losses['L2Decay_WeightsDepth'] = decay 
                decay = tf.multiply(tf.nn.l2_loss(self._weightsPoint), convWD, name=scope.name+'l2_wd_point')
                self._losses['L2Decay_WeightsPoint'] = decay 
        
            if bias: 
                self._bias = cpu_variable(scope.name+'_bias', [convChannels], \
                                          initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables['Bias'] = self._bias
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._bias), convWD, name=scope.name+'l2_bd')
                    self._losses['L2Decay_Bias'] = decay 
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = cpu_variable(scope.name+'_offset', \
                                             shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = cpu_variable(scope.name+'_scale', \
                                             shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = cpu_variable(scope.name+'_movMean', \
                                             shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = cpu_variable(scope.name+'_movVar', \
                                             shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables['BN_Offset'] = self._offset
                self._variables['BN_Scale'] = self._scale
                self._variables['BN_MovMean'] = self._movMean
                self._variables['BN_MovVar'] = self._movVar
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._scale)+tf.nn.l2_loss(self._offset), convWD, name=scope.name+'l2_bnd')
                    self._losses['L2Decay_BN'] = decay 
                
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                                  assign_moving_average(self._movVar, var, movingRate)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                 self._offset, self._scale, self._epsilon, \
                                                 name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            
            self._pool = pool
            if pool:
                self._sizePooling     = [1] + poolSize + [1]
                self._stridePooling   = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, \
                                  padding=self._typePoolPadding, name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
                
        self._output = self._outWrapper(pooled)
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
            
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeDepthKernel) + ', ' + str(self._sizePointKernel) + '; '  + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Activation: ' + activation + 'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + ']')

class DepthwiseConv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, movingRate=0.9, epsilon=1e-8, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 name=None, dtype=tf.float32): 
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        assert convChannels % feature.get_shape().as_list()[3] == 0, 'convChannels must be multiples of convChannels of the previous layer'
        
        Layer.__init__(self)
        
        self._type = 'DepthwiseConv2D'
        self._name = name
        
        with tf.variable_scope(self._name) as scope: 
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3], int(convChannels/feature.get_shape().as_list()[3])]
            self._strideConv      = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            
            self._weightsDepth = cpu_variable(scope.name+'_weights', \
                                              self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.depthwise_conv2d(feature, self._weightsDepth, strides=self._strideConv, padding=self._typeConvPadding, 
                                          name=scope.name+'_depthwise_conv')
            self._variables['Weights'] = self._weightsDepth
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), convWD, name=scope.name+'l2_wd_depth')
                self._losses['L2Decay_WeightsDepth'] = decay 
        
            if bias: 
                self._bias = cpu_variable(scope.name+'_bias', [convChannels], \
                                          initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables['Bias'] = self._bias
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._bias), convWD, name=scope.name+'l2_bd')
                    self._losses['L2Decay_Bias'] = decay 
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = cpu_variable(scope.name+'_offset', \
                                             shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = cpu_variable(scope.name+'_scale', \
                                             shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = cpu_variable(scope.name+'_movMean', \
                                             shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = cpu_variable(scope.name+'_movVar', \
                                             shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables['BN_Offset'] = self._offset
                self._variables['BN_Scale'] = self._scale
                self._variables['BN_MovMean'] = self._movMean
                self._variables['BN_MovVar'] = self._movVar
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._scale)+tf.nn.l2_loss(self._offset), convWD, name=scope.name+'l2_bnd')
                    self._losses['L2Decay_BN'] = decay 
                
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                                  assign_moving_average(self._movVar, var, movingRate)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                 self._offset, self._scale, self._epsilon, \
                                                 name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            
            self._pool = pool
            if pool:
                self._sizePooling     = [1] + poolSize + [1]
                self._stridePooling   = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, \
                                  padding=self._typePoolPadding, name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
                
        self._output = self._outWrapper(pooled)
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
            
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeDepthKernel) +  '; '  + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Activation: ' + activation + 'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + ']')



class DeConv2D(Layer):
    
    def __init__(self, feature, convChannels, shapeOutput=None, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, movingRate=0.9, epsilon=1e-8, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 name=None, dtype=tf.float32): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
                
        Layer.__init__(self)
        
        self._type = 'DeConv2D'
        self._name = name
        
        with tf.variable_scope(self._name) as scope: 
            self._sizeKernel      = convKernel + [convChannels, feature.get_shape().as_list()[3]]
            self._strideConv      = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            if shapeOutput is None: 
                self._shapeOutput = tf.TensorShape([feature.get_shape().as_list()[0], feature.get_shape().as_list()[1]*convStride[0], feature.get_shape().as_list()[2]*convStride[1], convChannels])
            else:
                self._shapeOutput = tf.TensorShape([feature.shape[0]] + shapeOutput + [convChannels])
            self._weights = cpu_variable(scope.name+'_weights', self._sizeKernel, \
                                         initializer=convInit, dtype=dtype)
            conv = tf.nn.conv2d_transpose(feature, self._weights, self._shapeOutput, self._strideConv, \
                                          padding=self._typeConvPadding, name=scope.name+'_conv2d_transpose')
            self._variables['Weights'] = self._weights
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), convWD, name=scope.name+'l2_wd')
                self._losses['L2Decay_Weights'] = decay 
        
            if bias: 
                self._bias = cpu_variable(scope.name+'_bias', [convChannels], \
                                          initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables['Bias'] = self._bias
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._bias), convWD, name=scope.name+'l2_bd')
                    self._losses['L2Decay_Bias'] = decay 
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = cpu_variable(scope.name+'_offset', \
                                             shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = cpu_variable(scope.name+'_scale', \
                                             shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = cpu_variable(scope.name+'_movMean', \
                                             shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = cpu_variable(scope.name+'_movVar', \
                                             shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables['BN_Offset'] = self._offset
                self._variables['BN_Scale'] = self._scale
                self._variables['BN_MovMean'] = self._movMean
                self._variables['BN_MovVar'] = self._movVar
                if convWD is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._scale)+tf.nn.l2_loss(self._offset), convWD, name=scope.name+'l2_bnd')
                    self._losses['L2Decay_BN'] = decay 
                
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                                  assign_moving_average(self._movVar, var, movingRate)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                 self._offset, self._scale, self._epsilon, \
                                                 name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            
            self._pool = pool
            if pool:
                self._sizePooling     = [1] + poolSize + [1]
                self._stridePooling   = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, \
                                  padding=self._typePoolPadding, name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
                
        self._output = self._outWrapper(pooled)
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
            
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeKernel) + '; ' + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Activation: ' + activation + ';' + 'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Stride: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + ']')
    
# Normalizations

class LRNorm(Layer):
    
    def __init__(self, feature, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'LRNorm'
        self._name        = name
        self._depthRadius = depth_radius
        self._bias        = bias
        self._alpha       = alpha
        self._beta        = beta
        with tf.variable_scope(self._name) as scope: 
            output = tf.nn.lrn(feature, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, \
                               name=scope.name)
            self._output = self._outWrapper(output)      
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + 'Output Size: ' + str(self._output.shape) + '; '  + \
                '; ' + 'Depth Radius: ' + self._depthRadius + '; ' + \
                'Bias: ' + self._bias + '; ' + 'Alpha: ' + self._alpha + '; ' + 'Beta: ' + self._beta + ']')
        
class BatchNorm(Layer):
    
    def __init__(self, feature, step, ifTest, movingRate=0.9, epsilon=1e-8, name=None, dtype=tf.float32): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self) 
        self._type = 'BatchNorm'
        self._name = name
        with tf.variable_scope(self._name) as scope: 
            shapeParams   = [feature.shape[-1]]
            self._offset  = cpu_variable(scope.name+'_offset', \
                                         shapeParams, initializer=ConstInit(0.0), dtype=dtype)
            self._scale   = cpu_variable(scope.name+'_scale', \
                                         shapeParams, initializer=ConstInit(1.0), dtype=dtype)
            self._movMean = cpu_variable(scope.name+'_movMean', \
                                         shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
            self._movVar  = cpu_variable(scope.name+'_movVar', \
                                         shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
            self._variables['BN_Offset'] = self._offset
            self._variables['BN_Scale'] = self._scale
            self._variables['BN_MovMean'] = self._movMean
            self._variables['BN_MovVar'] = self._movVar
            self._epsilon = epsilon
            def trainMeanVar(): 
                mean, var = tf.nn.moments(feature, list(range(len(feature.shape)-1)))
                with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                              assign_moving_average(self._movVar, var, movingRate)]): 
                    self._trainMean = tf.identity(mean)
                    self._trainVar  = tf.identity(var)
                return self._trainMean, self._trainVar
                
            self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
            self._output = tf.nn.batch_normalization(feature, self._actualMean, self._actualVar, \
                                                     self._offset, self._scale, self._epsilon, \
                                                     name=scope.name+'_batch_normalization')
            self._output = self._outWrapper(self._output)  
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Epsilon: ' + str(self._epsilon) + ']')
     
class Dropout(Layer): 
    
    def __init__(self, feature, ifTest, rateKeep=0.5, name=None): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = type 
        self._name = name
        self._rateKeep = rateKeep
        with tf.variable_scope(self._name) as scope:
            self._keepProb = cpu_variable(name=scope.name+'_keep_prob', shapeParams=[1], initializer=ConstInit(rateKeep), trainable=False, dtype=dtype)
            def phaseTest(): 
                return tf.assign(self._keepProb, 1.0)
            def phaseTrain(): 
                return tf.assign(self._keepProb, rateKeep)
            with tf.control_dependencies([tf.cond(ifTest, phaseTest, phaseTrain)]): 
                self._output = tf.nn.dropout(feature, self._keepProb, name=scope.name+'_dropout')
                self._output = self._outWrapper(self._output)  
                
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Keep Rate: ' + str(self._rateKeep) + ']')

# Fully Connected

class Flatten(Layer):
    def __init__(self, feature, name=None):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self); 
        self._type = 'Flatten'
        self._name = name
        size = feature.shape[1]
        for elem in feature.shape[2:]: 
            size *= elem
        self._output = tf.reshape(feature, [-1, size])
        self._output = self._outWrapper(self._output)  
        
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + ']')
    
class FullyConnected(Layer):
    
    def __init__(self, feature, outputSize, weightInit=XavierInit, wd=None, \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, movingRate=0.9, epsilon=1e-8, \
                 activation=ReLU, \
                 fakeQuant=False, name=None, dtype=tf.float32):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'FullyConnected'
        self._name = name
        with tf.variable_scope(self._name) as scope: 
            self._sizeWeights = [feature.get_shape().as_list()[1], outputSize]
            self._weights = cpu_variable(scope.name+'_weights', \
                                         self._sizeWeights, initializer=weightInit, dtype=dtype)
            if fakeQuant: 
                a = tf.reduce_min(self._weights)
                b = tf.reduce_max(self._weights)
                a = -tf.reduce_max(tf.abs(self._weights))
                b = tf.reduce_max(tf.abs(self._weights))
                self._weightMin = a; 
                self._weightMax = b; 
                self._weights = fake_quant_with_min_max_vars(self._weights, a, b, num_bits=FAKEBITS, narrow_range=False)
            self._variables['Weights'] = self._weights
            if wd is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), wd, name=scope.name+'l2_wd')
                # decay = tf.multiply(tf.reduce_sum(tf.abs(self._weights)), wd, name=scope.name+'l1_wd')
                self._losses['L2Decay_Weights'] = decay
            if bias: 
                self._bias = cpu_variable(scope.name+'_bias', [outputSize], \
                                          initializer=biasInit, dtype=dtype)
                self._variables['Bias'] = self._bias
                # if fakeQuant: 
                #     a = tf.reduce_min(self._bias)
                #     b = tf.reduce_max(self._bias)
                #     a = -tf.reduce_max(tf.abs(self._bias))
                #     b = tf.reduce_max(tf.abs(self._bias))
                #     self._biasMin = a; 
                #     self._biasMax = b; 
                #     self._bias = fake_quant_with_min_max_vars(self._bias, a, b, num_bits=FAKEBITS, narrow_range=False)
                if wd is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._bias), wd, name=scope.name+'l2_bd')
                    # decay = tf.multiply(tf.reduce_sum(tf.abs(self._bias)), wd, name=scope.name+'l1_bd')
                    self._losses['L2Decay_Bias'] = decay 
            else:
                self._bias = tf.constant(0.0, dtype=dtype)
            
            output = tf.add(tf.matmul(feature, self._weights), self._bias, name=scope.name+'_fully_connected')
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [output.shape[-1]]
                self._offset  = cpu_variable(scope.name+'_offset', \
                                             shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = cpu_variable(scope.name+'_scale', \
                                             shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = cpu_variable(scope.name+'_movMean', \
                                             shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = cpu_variable(scope.name+'_movVar', \
                                             shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables['BN_Offset'] = self._offset
                self._variables['BN_Scale'] = self._scale
                self._variables['BN_MovMean'] = self._movMean
                self._variables['BN_MovVar'] = self._movVar
                if wd is not None:
                    decay = tf.multiply(tf.nn.l2_loss(self._scale)+tf.nn.l2_loss(self._offset), wd, name=scope.name+'l2_bnd')
                    # decay = tf.multiply(tf.reduce_sum(tf.abs(self._scale))+tf.reduce_sum(tf.abs(self._offset)), wd, name=scope.name+'l1_bnd')
                    self._losses['L2Decay_BN'] = decay 
                
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(output, list(range(len(output.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, movingRate), \
                                                  assign_moving_average(self._movVar, var, movingRate)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                output = tf.nn.batch_normalization(output, self._actualMean, self._actualVar, \
                                                   self._offset, self._scale, self._epsilon, \
                                                   name=scope.name+'_batch_normalization')
            self._output = output
            self._activation = activation 
            if activation is not None:
                self._output = activation(self._output, name=scope.name+'_activation')
            
            self._output = self._outWrapper(self._output)  
            
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Weight Size: ' + str(self._weights.shape) + '; ' + \
                'Bias Size: ' + str(self._bias.shape) + '; ' + \
                'Activation: ' + activation + ']')
        
class Activation(Layer):
    
    def __init__(self, feature, activation=Linear, name=None):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._name = name
        self._type = 'Activation'
        self._activation = activation
        with tf.variable_scope(self._name) as scope: 
            self._output = activation(feature, name=scope.name+'_activation')
            self._output = self._outWrapper(self._output)  
        
    @property
    def summary(self): 
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Activation: ' + activation + ']')
        
class Pooling(Layer):
    
    def __init__(self, feature, pool=MaxPool, size=[2, 2], stride=[2, 2], padding='SAME', name=None):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'Pooling'
        self._name = name
        self._typePool = pool
        self._sizePooling     = [1] + size + [1]
        self._stridePooling   = [1] + stride + [1]
        self._typePoolPadding = padding
        with tf.variable_scope(self._name) as scope: 
            self._output = self._typePool(feature, ksize=self._sizePooling, strides=self._stridePooling, \
                                          padding=self._typePoolPadding, \
                                          name=scope.name+'_pooling')
            self._output = self._outWrapper(self._output)  
    @property
    def summary(self): 
        if isinstance(self._typePool, functools.partial): 
            pooltype = self._typePool.func.__name__
        else:
            pooltype = self._typePool.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Type: ' + pooltype + ']')
        
class GlobalAvgPool(Layer):
    
    def __init__(self, feature, name=None):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'GlobalAvgPool'
        self._name = name
        with tf.variable_scope(self._name) as scope: 
            self._output = tf.reduce_mean(feature, [1, 2], name=scope.name+'_global_avg_pool')
            self._output = self._outWrapper(self._output)  
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Type: Global Average Pooling' + ']')
        
class CrossEntropy(Layer):
    
    def __init__(self, feature, labels, \
                 name=None):
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'CrossEntropy'
        self._name = name
        with tf.variable_scope(self._name) as scope: 
            self._output = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=feature, \
                                                                          name=scope.name+'_cross_entropy') 
            self._output = tf.reduce_mean(self._output)
            self._losses['CrossEntropyLoss'] = self._output
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Activation: CrossEntropy' + ']')

class TripletLoss(Layer):
    def __init__(self, feature, numDiff=1, weightDiff=1.0, tao=-1.0,  name=None): 
        assert not (name is None), 'name must not be None'
        
        if not isinstance(feature, tf.Tensor): 
            if hasattr(feature, 'output'): 
                feature = feature.output
        assert isinstance(feature, tf.Tensor), 'feature or feature.output must be a tf.Tensor'
        
        Layer.__init__(self)
        self._type = 'TripletLoss'
        self._name = name
        self._numDiff = numDiff
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name) as scope: 
            numPerGroup = 2 + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossDiff = 0.0
            for idx in range(2, 2+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(tf.maximum(self._lossPos - self._weightDiff*self._lossDiff, tao), name=scope.name+'_multilet_loss')
            self._losses['TripletLoss'] = self._output
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: TripletLoss' + ']')

class TripletAccu(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self)
        self._type = 'TripletAccu'
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        with tf.variable_scope(self._name) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1)
            accum = 0.0
            for idx in range(2, 2+numSame):  
                accum = accum + tf.reduce_mean(tf.cast(tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                                  tf.zeros_like(self._lossPos)), tf.float32))
            for idx in range(2+numSame, 2+numSame+numDiff):  
                accum = accum + tf.reduce_mean(tf.cast(tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                                  tf.zeros_like(self._lossPos)), tf.float32))
            self._output = tf.identity(accum/(numSame + numDiff), name=scope.name+'_accu')
            
    @property
    def type(self):
        return 'TripletAccu'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: TripletAccu' + ']')

