####################################################################
# Test 1D: Black-Scholes
####################################################################
import sys
sys.path.append("./")
sys.path.append("../")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.fnn import FNN
from src.spvsd import Spvsd
from src.losses import spvsd_loss, BS_SPINN, spvsd_gradients_loss
from scipy.special import ndtr

def blackScholesMerton(
    S:  np.array,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    flag="c"
    ):
    """
    Return the Black-Scholes-Merton option price
    :param S: stock value, S >= 0
    :param strike: fixed strike, K >= 0
    :param tau: time to maturity
    :param r: risk-free interest rate
    :param sigma: volatility
    :param q: some kind of dividend yield
    :param phi: 1 for call, -1 for put
    :return: Black Scholes value for the given inputs
    """
    if flag == "c":
        alpha = 1
    elif flag == "p":
        alpha = -1
    else:
        raise ValueError("flag must be c for call or p for put!")
    
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    V = alpha * S * np.exp(- q * tau) * ndtr(alpha * d1) \
        - alpha * strike * np.exp(-r * tau) * ndtr(alpha * d2)

    return V

def blackScholesTheta(
    S:  np.array,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    flag="c"
    ):
    """
    Return the Black-Scholes-Merton option price
    :param S: stock value, S >= 0
    :param strike: fixed strike, K >= 0
    :param tau: time to maturity
    :param r: risk-free interest rate
    :param sigma: volatility
    :param q: some kind of dividend yield
    :param phi: 1 for call, -1 for put
    :return: Black Scholes value for the given inputs
    """
    if flag == "c":
        alpha = 1
    elif flag == "p":
        alpha = -1
    else:
        raise ValueError("flag must be c for call or p for put!")
    
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    V = -S*sigma/(2.*np.sqrt(tau))*np.exp(-d1*d1/2)/(np.sqrt(2*np.pi)) \
        - alpha * r * strike*np.exp(-r * tau) * ndtr(alpha * d2)

    return V

def blackScholesDelta(
    S:  np.array,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    flag="c"
    ):
    """
    Return the Black-Scholes-Merton option price
    :param S: stock value, S >= 0
    :param strike: fixed strike, K >= 0
    :param tau: time to maturity
    :param r: risk-free interest rate
    :param sigma: volatility
    :param q: some kind of dividend yield
    :param phi: 1 for call, -1 for put
    :return: Black Scholes value for the given inputs
    """
    if flag == "c":
        alpha = 1
    elif flag == "p":
        alpha = -1
    else:
        raise ValueError("flag must be c for call or p for put!")
    
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return alpha*ndtr(alpha * d1)

# Gemerate dataset
n_points = int(1e2)
S = np.linspace(0.5,1.5,n_points)[:,None]
tau = np.linspace(0.1,1.5,n_points)[:,None]
X1, X2 = np.meshgrid(tau, S)
x = np.vstack((np.ravel(X1), np.ravel(X2))).T

K = 1.

r = 0.01
sigma = 0.25

y = blackScholesMerton(x[:,1],K,x[:,0],r,sigma,0.0,"p")[:,None]
y_grad_0 = -blackScholesTheta(x[:,1],K,x[:,0],r,sigma,0.0,"p")[:,None]
y_grad_1 = blackScholesDelta(x[:,1],K,x[:,0],r,sigma,0.0,"p")[:,None]

target = np.concatenate((y,y_grad_0,y_grad_1),axis = 1)

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

initial_learning_rate = 0.05
decay_steps = 100
decay_rate = 0.5
learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate, decay_steps, decay_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
epochs = 2000

#####################
# Spvsd
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate, decay_steps, decay_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
# Compilation
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

mask_metric = [0,1,2]
loss = lambda func, x, y: spvsd_loss(func,x,y,tf.keras.losses.MeanSquaredError())

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)

# Training
metric_history_1 = spvsd.fit(x_train,y_train,epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_1 = spvsd.call(tf.constant(x,dtype = tf.float32))

#####################
# Spvsd+SPINN
####################
net = FNN([n_inputs]+[n_units]*n_layers+[n_outputs],activation = activation ,kernel_init = kernel_init)
spvsd = Spvsd(net)
learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate, decay_steps, decay_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
# Compilation
loss = tf.keras.losses.MeanSquaredError()
metric_1 = tf.keras.metrics.MeanSquaredError()
metric_2 = tf.keras.metrics.MeanAbsoluteError()
metrics = [metric_1,metric_2]

mask_metric = [0,1,2]
loss_weights = [0.5,0.5]
loss = lambda func, x, y: BS_SPINN(func,x,y,loss_weights,tf.keras.losses.MeanSquaredError(),r,sigma)

spvsd.compile(optimizer,loss,metrics, mask = mask_metric)

# Training
metric_history_2 = spvsd.fit(x_train,y_train[:,0][:,None],epochs,validation_data = (x_test,y_test))

# Prediction
y_pred_2 = spvsd.call(tf.constant(x,dtype = tf.float32))




####################
# Plot
####################
fig, ax = plt.subplots(nrows = 1, ncols = 2 )
ax[0].plot(np.arange(epochs),metric_history_1[1:,0,0].astype(np.float32) , color = "green", label = "Spvsd" )
ax[0].plot(np.arange(epochs),metric_history_2[1:,0,0].astype(np.float32), color = "pink", label = "Spvsd+SPINN" )
ax[0].set_title("MSE")
ax[0].grid()
ax[0].set_yscale("log")
plt.legend()


ax[1].plot(x[:,1], y[:,0], color = "blue", label = "Exact" )
ax[1].plot(x[:,1], y_pred_1[:,0], color = "green", label = "Spvsd" )
ax[1].plot(x[:,1], y_pred_2[:,0], color = "pink", label = "Spvsd+SPINN" )
ax[1].set_title("Function")
ax[1].grid()
plt.legend()

plt.show()
