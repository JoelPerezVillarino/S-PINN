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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

spvsd.compile(optimizer,loss,metrics)

# Training
epochs = 100
x1 = np.linspace(0,np.pi)[:,None]
x2 = np.linspace(0,np.pi)[:,None]
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack((np.ravel(X1), np.ravel(X2))).T
y = (np.sin(x[:,0]) + np.sin(x[:,1]))[:,None]
y_grad_0 = (np.cos(x[:,0]))[:,None]
y_grad_1 = (np.cos(x[:,1]))[:,None]
target = np.concatenate((y,y_grad_0,y_grad_1), axis = 1)
spvsd.fit(x,target,epochs)

# Prediction
y_pred = spvsd.predict(x)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.scatter(x[:,0],x[:,1], y_pred[:,0], color = "blue", label = "net" )
surf2 = ax.scatter(x[:,0], x[:,1], y[:,0], color = "red", label = "exact")
plt.legend()
plt.show()
