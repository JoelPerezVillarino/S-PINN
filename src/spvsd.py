import numpy as np
import tensorflow as tf
from itertools import compress
from src.losses import spvsd_loss


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

        self.compute_loss = None

    def compile(self, optimizer = None, loss = None, metrics = None,  mask = None,):
        print("Compiling model...")
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(0.01)
        self.opt = optimizer 

        if loss is None: 
            loss = lambda func, x, y: spvsd(func, x, y, tf.keras.losses.MeanSquaredError())
        self.loss = loss
        
        if type(metrics) != list:
            metrics = [metrics]
        self.metrics = metrics 
        
        
        if mask is None:
            mask = [ i for i in range(self.net.n_inputs+1+int(self.net.n_inputs**2)) ]
        self.mask = mask

        self.metric_history = []
       
        # Initialize metric history
        epoch_list = []
        for i in range(len(self.metrics)):
            metrics_list = []
            for j in range(len(self.mask)):
                string = self.metrics[i].name+" "+str(j)
                metrics_list.append(string)
            epoch_list.append(metrics_list)

        self.metric_history.append(epoch_list)


    
    def compute_metrics(self,validation_data):
        
        x_test = validation_data[0]
        y_test = validation_data[1]
        y_pred = self.call(x_test)
        y_pred = tf.gather(y_pred,self.mask, axis = 1) 
        
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
            #y_pred = self.call(x,training = True)
            total_loss = self.loss(self.call,x,y)
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
            

        return np.array(self.metric_history)

    
    def call2(self, x, training = None):
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y = self.net(x,training = training)
            y_grad = tape2.gradient(y,x)
        y_hess = tape1.gradient(y_grad,x)
        result = tf.concat((y,y_grad,y_hess),axis = 1)

        # Erase tapes
        del tape1
        del tape2
        
        return result
    
    def call(self, x, training = None):
        with tf.GradientTape(persistent = True) as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y = self.net(x,training = training)
            y_grad = tape2.gradient(y,x)
            y_grad_unstack = tf.unstack(y_grad,axis = 1)
        d2f_dx2 = []
        for df_dx in y_grad_unstack:
            d2f_dx2.append(tape1.gradient(y_grad_unstack,x))
        y_hess = tf.concat(d2f_dx2, axis = 1)
        result = tf.concat((y,y_grad,y_hess),axis = 1)

        # Erase tapes
        del tape1
        del tape2
        
        return result
    

