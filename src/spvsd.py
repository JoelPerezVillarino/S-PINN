import numpy as np
import tensorflow as tf
from itertools import compress


class Spvsd(object):

    def __init__(self,  net, ):
        self.net = net

        self.opt = None
        self.metrics = None

        # methods initialized in compile
        self.loss = None
        self.loss_weights = None
        self.metric_weights = None
        self.mask = None

        # History
        self.metric_history = None


    def compile(self, optimizer = None, loss = None, metrics = None, loss_weights=None, metric_weights = None, mask = None):
        print("Compiling model...")
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(0.01)
        self.opt = optimizer 

        if loss is None: 
            loss = tf.keras.losses.MeanSquaredError()
        self.loss = loss
        
        if type(metrics) != list:
            metrics = [metrics]
        self.metrics = metrics 
        
        if loss_weights is None:
            loss_weights = tf.convert_to_tensor([0.33, 0.33, 0.33])
        self.loss_weights = loss_weights
        
        if metric_weights is None:
            metric_weights = tf.convert_to_tensor([1., 0., 0.])
        self.metric_weights = metric_weights
        
        if mask is None:
            mask = [ 1 for i in range(self.net.n_inputs+1) ]
        self.mask = mask

        self.metric_history = []
        
        epoch_list = []
        for i in range(len(self.metrics)):
            metrics_list = []
            for j in range(len(self.mask)):
                string = self.metrics[i].name+" "+str(j)
                metrics_list.append(string)
            epoch_list.append(metrics_list)

        self.metric_history.append(epoch_list)


    def compute_loss(self,y,y_pred):
        losses = [self.loss(y[:,i],y_pred[:,i]) for i in range(y_pred.shape[1])]
        losses = tf.convert_to_tensor(losses)
        # Weighted losses
        losses *= self.loss_weights
        total_loss = tf.math.reduce_sum(losses)
        return total_loss
    
    def compute_metrics(self,validation_data):
        x_test = validation_data[0]
        y_test = validation_data[1]
        y_pred = self.call(x_test)
        
        title = [ '{0: <25}'.format(str(i)) for i in range(sum(self.mask)) ]
        underline = [ '{0: <25}'.format("-"*25) for i in range(sum(self.mask)) ]
        title = ''.join(title)
        underline = ''.join(underline)
        print('{0: <25}'.format(" ")+title)
        print('{0: <25}'.format(" ")+underline)
        
        epoch_list = []
        for i in range(len(self.metrics)):
            metric_list = []
            metric_string = []
            metric = self.metrics[i]
            for j in range(y_pred.shape[1]):
                metric.update_state(y_test[:,j],y_pred[:,j])
                result = float(self.metrics[i].result())
                metric.reset_state()
                metric_list.append(result)
                metric_string.append('{0: <25}'.format(str(result)))
            
            print('{0: <25}'.format(metric.name)+''.join(metric_string))
            epoch_list.append(metric_list)

        print("\n")
        self.metric_history.append(epoch_list)

        return None
    
    @tf.function
    def train_step(self,data):
        x = data[0]
        y = data[1]
        with tf.GradientTape() as tape:
            y_pred = self.call(x,training = True)
            total_loss = self.compute_loss(y,y_pred)
        grads = tape.gradient(total_loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))



    def fit(self, x, y, epochs=100, validation_data=None,display_every=1):
        print("Training model...\n")
        x_train = tf.constant(x, dtype = tf.float32)
        y_train = tf.constant(y, dtype = tf.float32)
        data = (x_train,y_train)

        x_test = tf.constant(validation_data[0], dtype = tf.float32)
        y_test = tf.constant(validation_data[1], dtype = tf.float32)
        validation_data = (x_test,y_test)


        for i in range(epochs):
            self.train_step(data)
            print("Epoch: ",i+1)
            self.compute_metrics(validation_data)
            

        return self.metric_history 

    
    def call(self, x, training = None):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.net(x,training = training)
        y_grad = tape.gradient(y,x)
        y_concat = tf.concat((y,y_grad), axis = 1)
        y_mask = tf.gather(y_concat,self.mask, axis = 1) 
        return y_mask
    

