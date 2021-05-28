import tensorflow as tf

from tensorflow.python.training.moving_averages import assign_moving_average

class Layer: 
    
    def __init__(self):
        self._variables = {}
        self._losses = {}
        self._updateOps = {}
        self._name = None
        self._type = 'Layer'
        self._output = None
        self._outMax = None
        self._outMin = None
        self._fakeQuant = False
    
    def _outWrapper(self, output): 
        movingRate = 0.9
        self._outMinTmp = tf.reduce_min(output)
        self._outMaxTmp = tf.reduce_max(output)
        self._outMin    = tf.Variable(0.0, trainable=False)
        self._outMax    = tf.Variable(0.0, trainable=False)
        self._assignMin = assign_moving_average(self._outMin, self._outMinTmp, movingRate)
        self._assignMax = assign_moving_average(self._outMax, self._outMaxTmp, movingRate)
        with tf.control_dependencies([self._assignMin, self._assignMax]): 
            output = tf.identity(output, name="FinalOutput")
        return output
    
    def setMinMax(self, actMin, actMax): 
        with tf.control_dependencies([self._assignMin, self._assignMax]): 
            with tf.control_dependencies([tf.assign(self._outMin, actMin), tf.assign(self._outMax, actMax)]): 
                self._output = tf.identity(self._output)
    
    @property
    def type(self):  
        return self._type
    
    @property
    def name(self):  
        return self._name
    
    @property
    def output(self):  
        return self._output
    
    @property
    def outMin(self): 
        return self._outMin
    
    @property
    def outMax(self):  
        return self._outMax
    
    @property
    def variables(self):
        return self._variables
    
    @property
    def losses(self):
        return self._losses
    
    @property
    def updateOps(self):
        return self._updateOps
    
    @property
    def summary(self):
        return 'Layer: the parent class of all layers'

class Net:
    
    def __init__(self, HParam, name):
        
        self._layers    = []
        self._updateOps = []
        self._body      = None
        self._inference = None
        self._loss      = None
        
        self._init   = False
        self._HParam = HParam
        self._name   = name
        self._graph  = tf.Graph()
        self._sess   = tf.Session(graph=self._graph)
        
        with self._graph.as_default(), tf.device('/cpu:0'), tf.variable_scope(self._name, reuse=tf.AUTO_REUSE): 
        
            self._ifTest        = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step          = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain    = tf.assign(self._ifTest, False)
            self._phaseTest     = tf.assign(self._ifTest, True)
            
            self._lr = tf.reduce_max([tf.train.exponential_decay(self._HParam['LearningRate'], global_step=self._step, decay_steps=self._HParam['LRDecayAfter'], decay_rate=self._HParam['LRDecayRate']), self._HParam['MinLearningRate']])    
        
    def save(self, path):
        pass
        
    def load(self, path):
        pass
    
    def getLoss(self, layers=None):
        
        loss = 0
        
        if layers is None: 
            for elem in self._layers: 
                if len(elem.losses.keys()) > 0: 
                    for tmp in elem.losses.keys(): 
                        loss += elem.losses[tmp]
        else: 
            for elem in layers: 
                if len(elem.losses.keys()) > 0: 
                    for tmp in elem.losses.keys(): 
                        loss += elem.losses[tmp]
            
        return loss
    
    def getUpdateOps(self, layers=None): 
        
        updateOps = []
        
        if layers is None: 
            for elem in self._layers: 
                if len(elem.updateOps) > 0: 
                    for tmp in elem.updateOps.keys(): 
                        updateOps.append(elem.updateOps[tmp])
        else:
            for elem in layers: 
                if len(elem.updateOps) > 0: 
                    for tmp in elem.updateOps.keys(): 
                        updateOps.append(elem.updateOps[tmp])
            
        return updateOps
    
    def getLR(): 
        
        return self._sess.run(self._lr)
    
    def setLR(value): 
        
        self._sess.run(tf.assign(self._lr, value))
    
    @property
    def summary(self): 
    
        summs = []
        summs.append("=>Network Summary: ")
        for elem in self._layers:
            summs.append(elem.summary)
        summs.append("<=Network Summary. ")
        
        return "\n\n".join(summs)
    

