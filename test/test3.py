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
from src.losses import spvsd_loss, exp_grad_loss, exp_hess_loss, exp_grad_hess_loss, spvsd_gradients_loss

# Gemerate dataset
n_points = int(1e5)
x = np.linspace(0,1,n_points)[:,None]
y = np.exp(x) 
y_grad = (np.exp(x))
y_grad_grad = (np.exp(x))
target = np.concatenate((y,y_grad,y_grad_grad), axis = 1)
print("Mean: ", np.mean(y))

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
n_inputs = 1
n_outputs = 1
n_layers = 3
n_units = 4

activation = "tanh"
kernel_init = "glorot_uniform"



#####################
# Values
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

mask_metric = [0,1,2]
loss = lambda func, x, y: spvsd_loss(func,x,y,tf.keras.losses.MeanSquaredError())

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)

# Training
epochs = 300
metric_history_1 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_1 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# SPINN gradient
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]


mask_metric = [0,1,2]
loss_weights = [0.5, 0.5]
loss = lambda func, x, y: exp_grad_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError())

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)
# Training
epochs = 300
metric_history_2 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_2 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# SPINN hessian
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]


mask_metric = [0,1,2]
loss_weights = [0.5, 0.5]
loss = lambda func, x, y: exp_hess_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError())

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)
# Training
epochs = 300
metric_history_3 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_3 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# SPINN gradient+hessian
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]


mask_metric = [0,1,2]
loss_weights = [0.33, 0.33, 0.33]
loss = lambda func, x, y: exp_grad_hess_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError())

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)
# Training
epochs = 300
metric_history_4 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_4 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# Spvsd gradient+hessian
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.1)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]


mask_metric = [0,1,2]
mask_loss = [0,2]
loss_weights = [0.5, 0.5]
loss = lambda func, x, y: spvsd_gradients_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError(), mask = mask_loss)

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)
# Training
epochs = 300
metric_history_5 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_5 = spvsd.call(tf.constant(x,dtype = tf.float32))




####################
# Plot
####################
fig, ax = plt.subplots(nrows = 1, ncols = 3 )
ax[0].plot(x[:], np.abs(y_pred_1[:,0]-y[:,0]), color = "blue", label = "Spvsd value" )
ax[0].plot(x[:], np.abs(y_pred_2[:,0]-y[:,0]), color = "red", label = "SPINN value+grad" )
ax[0].plot(x[:], np.abs(y_pred_3[:,0]-y[:,0]), color = "green", label = "SPINN value+hess" )
ax[0].plot(x[:], np.abs(y_pred_4[:,0]-y[:,0]), color = "black", label = "SPINN value+grad+hess" )
ax[0].plot(x[:], np.abs(y_pred_5[:,0]-y[:,0]), color = "pink", label = "Spvsd value+hess" )
ax[0].set_title("Error")
ax[0].grid()
plt.legend()

ax[1].plot(np.arange(epochs),metric_history_1[1:,0,0].astype(np.float32) , color = "blue", label = "Spvsd value" )
ax[1].plot(np.arange(epochs),metric_history_2[1:,0,0].astype(np.float32), color = "red", label = "SPINN value+grad" )
ax[1].plot(np.arange(epochs),metric_history_3[1:,0,0].astype(np.float32), color = "green", label = "SPINN value+hess" )
ax[1].plot(np.arange(epochs),metric_history_4[1:,0,0].astype(np.float32), color = "black", label = "SPINN value+grad+hess" )
ax[1].plot(np.arange(epochs),metric_history_5[1:,0,0].astype(np.float32) , color = "pink", label = "Spvsd value+hess" )
ax[1].set_title("MSE")
ax[1].grid()
ax[1].set_yscale("log")
plt.legend()

ax[2].plot(x[:], np.abs(y_pred_1[:,0]), color = "blue", label = "Spvsd value" )
ax[2].plot(x[:], np.abs(y_pred_2[:,0]), color = "red", label = "SPINN value+grad" )
ax[2].plot(x[:], np.abs(y_pred_3[:,0]), color = "green", label = "SPINN value+hess" )
ax[2].plot(x[:], np.abs(y_pred_4[:,0]), color = "black", label = "SPINN value+grad+hess" )
ax[2].plot(x[:], np.abs(y_pred_5[:,0]), color = "pink", label = "Spvsd value+hess" )
plt.legend()
ax[2].set_title("Function")
ax[2].grid()

plt.show()
