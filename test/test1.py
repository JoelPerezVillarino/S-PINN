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
from src.losses import spvsd_loss, spvsd_gradients_loss

# Gemerate dataset
x1 = np.linspace(0,np.pi)[:,None]
x2 = np.linspace(0,np.pi)[:,None]
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack((np.ravel(X1), np.ravel(X2))).T
y = (np.sin(x[:,0]) + np.sin(x[:,1]))[:,None]

y_grad_0 = (np.cos(x[:,0]))[:,None]
y_grad_1 = (np.cos(x[:,1]))[:,None]

y_00 = (-np.sin(x[:,0]))[:,None]
y_01 = np.zeros_like(y_00)
y_10 = np.zeros_like(y_00)
y_11 = (-np.sin(x[:,1]))[:,None]

target = np.concatenate((y,y_grad_0,y_grad_1,y_00,y_01,y_10,y_11), axis = 1)

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
n_units = 10

activation = "tanh"
kernel_init = "glorot_uniform"


#####################
# Values+Derivatives
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.01)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

mask_metric = [0,1,2,3,4,5,6]
mask_loss = [0,3]
loss_weights = [0.5,0.5]
loss = lambda func, x, y: spvsd_gradients_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError(), mask_loss)

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)

# Training
epochs = 1000
metric_history_1 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_1 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# Values
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
# Compilation
optimizer = tf.keras.optimizers.Adam(lr = 0.01)
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

mask_metric = [0,1,2,3,4,5,6]
mask_loss = [0]
loss_weights = [1]
loss = lambda func, x, y: spvsd_gradients_loss(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError(), mask_loss)

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)

# Training
epochs = 1000
metric_history_2 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_2 = spvsd.call(tf.constant(x,dtype = tf.float32))


####################
# Plot
####################
#fig = plt.figure()
#ax1 = fig.add_subplot(221, projection='3d')
#ax1.scatter(x[:,0],x[:,1], target[:,0], color = "blue", label = "Exact" )
#ax1.scatter(x[:,0],x[:,1], y_pred_1[:,0], color = "red", label = "Spvsd+SPINN" )
#ax1.scatter(x[:,0],x[:,1], y_pred_2[:,0], color = "green", label = "Spvsd" )
#plt.legend()
#ax1.set_title("Values")
#
#ax2 = fig.add_subplot(222, projection='3d')
#ax2.scatter(x[:,0],x[:,1], target[:,3], color = "blue", label = "Exact" )
#ax2.scatter(x[:,0],x[:,1], y_pred_1[:,3], color = "red", label = "Spvsd+SPINN" )
#ax2.scatter(x[:,0],x[:,1], y_pred_2[:,3], color = "green", label = "Spvsd" )
#plt.legend()
#ax1.set_title("Second derivative")

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax1.plot(np.arange(epochs),metric_history_1[1:,0,0].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax1.plot(np.arange(epochs),metric_history_2[1:,0,0].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax1.set_title("Values")
ax1.set_yscale("log")

ax2 = fig.add_subplot(232)
ax2.plot(np.arange(epochs),metric_history_1[1:,0,1].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax2.plot(np.arange(epochs),metric_history_2[1:,0,1].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax2.set_title("Gradient 1")
ax2.set_yscale("log")

ax3 = fig.add_subplot(233)
ax3.plot(np.arange(epochs),metric_history_1[1:,0,2].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax3.plot(np.arange(epochs),metric_history_2[1:,0,2].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax3.set_title("Gradient 2")
ax3.set_yscale("log")

ax4 = fig.add_subplot(234)
ax4.plot(np.arange(epochs),metric_history_1[1:,0,3].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax4.plot(np.arange(epochs),metric_history_2[1:,0,3].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax4.set_title("Hess 11")
ax4.set_yscale("log")

ax5 = fig.add_subplot(235)
ax5.plot(np.arange(epochs),metric_history_1[1:,0,4].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax5.plot(np.arange(epochs),metric_history_2[1:,0,4].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax5.set_title("Hess cross")
ax5.set_yscale("log")

ax6 = fig.add_subplot(236)
ax6.plot(np.arange(epochs),metric_history_1[1:,0,6].astype(np.float32) , color = "red", label = "Spvsd+SPINN" )
ax6.plot(np.arange(epochs),metric_history_2[1:,0,6].astype(np.float32) , color = "green", label = "Spvsd" )
plt.legend()
ax6.set_title("Hess 22")
ax6.set_yscale("log")

plt.show()
