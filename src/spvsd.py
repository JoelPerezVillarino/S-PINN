import numpy as np
import tensorflow as tf


class Spvsd(object):

    def __init__(self,  net, ):
        self.net = net

        self.opt = None
        self.metrics = None

        # methods initialized in compile
        self.loss = None
        self.loss_weights = None


    def compile(self, optimizer, loss = None, metrics = None, loss_weights=None):
        print("Compiling model...")
        self.opt = optimizer 
        if type(metrics) != list:
            metrics = [metrics]
        self.metrics = metrics 

        if loss_weights is None:
            loss_weights = tf.convert_to_tensor([1.])
        self.loss_weights = loss_weights
       
        if loss is None: 
            loss = tf.keras.losses.MeanSquaredError()
        self.loss = loss


    def compute_loss(self,y,y_pred):
        losses = [self.loss(y[:, i],y_pred[:, i]) for i in range(y.shape[1])]
        losses = tf.convert_to_tensor(losses)
        # Weighted losses
        losses *= self.loss_weights
        total_loss = tf.math.reduce_sum(losses)
        return total_loss
    
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
        data = (x,y)

        for i in range(epochs):
            self.train_step(data)
            print(f"Epoch: {i}")

        return None 


    def predict(self, x):
        return self.net(x,training = False)
    
    def call(self, x, training = None):
        return self.net(x,training = training)
    

