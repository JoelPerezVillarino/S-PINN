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

# Gemerate dataset
x1 = np.linspace(0,np.pi)[:,None]
x2 = np.linspace(0,np.pi)[:,None]
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack((np.ravel(X1), np.ravel(X2))).T
y = (np.sin(x[:,0]) + np.sin(x[:,1]))[:,None]
y_grad_0 = (np.cos(x[:,0]))[:,None]
y_grad_1 = (np.cos(x[:,1]))[:,None]
target = np.concatenate((y,y_grad_0,y_grad_1), axis = 1)

# Train test split
test_size = 0.2
cut = int(x.shape[0]*(1-test_size))

idx = np.arange(x.shape[0])
np.random.shuffle(idx)
idx_train = idx[:cut]
idx_test = idx[cut:]

x_train = x[idx_train]
y_train = target[idx_train]
x_test = x[idx_test]
y_test = target[idx_test]

# Model parameters
n_inputs = 2
n_outputs = 1
n_layers = 3
n_units = 4

activation = "tanh"
kernel_init = "glorot_uniform"

net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)

spvsd = Spvsd(net)

#####################
# Values+Derivatives
####################
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]
mask = [0,1,2]
loss_weights = [0.33,0.33,0.33]

spvsd.compile(optimizer,loss,metrics, mask = mask, loss_weights = loss_weights)

# Training
epochs = 100
spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_1 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# Values+Derivatives
####################
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]
mask = [0]
loss_weights = [1]

spvsd.compile(optimizer,loss,metrics, mask = mask, loss_weights = loss_weights)

# Training
epochs = 100
spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_2 = spvsd.call(tf.constant(x,dtype = tf.float32))


####################
# Plot
####################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.scatter(x[:,0],x[:,1], np.abs(y_pred_1[:,0]-y[:,0]), color = "blue", label = "Error value+grad" )
surf2 = ax.scatter(x[:,0],x[:,1], np.abs(y_pred_2[:,0]-y[:,0]), color = "red", label = "Error value" )
plt.legend()
plt.show()
