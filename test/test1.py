####################################################################
# Test on spvsd
####################################################################
import sys
sys.path.append("./")
sys.path.append("../")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import numpy as np
import tensorflow as tf

from src.fnn import FNN
from src.spvsd import Spvsd

# Model parameters
n_inputs = 2
n_outputs = 1
n_layers = 3
n_units = 4

activation = "tanh"
kernel_init = "glorot_uniform"

net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)

spvsd = Spvsd(net)

# Compilation
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

spvsd.compile(optimizer,loss,metrics)

# Training
epochs = 10
x1 = np.linspace(0,np.pi)[:,None]
x2 = np.linspace(0,np.pi)[:,None]
x = np.concatenate((x1,x2),axis = 1)
print(x.shape)
y = (np.sin(x[:,0]) + np.sin(x[:,1]))[:,None]
spvsd.fit(x,y,epochs)


