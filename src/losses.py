import tensorflow as tf

def spvsd_gradients_loss(func,x,y, loss_weights, metric, mask):
    y_pred = func(x)
    y_pred = tf.gather(y_pred, mask, axis = 1)
    losses = [metric(y[:,i],y_pred[:,i]) for i in range(y_pred.shape[1])]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

def spvsd_loss(func,x,y, metric):
    y_pred = func(x)
    losses = metric(y[:,0],y_pred[:,0])
    losses = tf.convert_to_tensor(losses)
    return losses

# Exponential
def exp_grad_loss(func,x,y, loss_weights, metric):
    y_pred = func(x)
    loss_spvsd = metric(y[:,0],y_pred[:,0]) 
    loss_pinn = metric(y[:,0],y_pred[:,1]) 
    losses = [loss_spvsd, loss_pinn]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

def exp_hess_loss(func,x,y, loss_weights, metric):
    y_pred = func(x)
    loss_spvsd = metric(y[:,0],y_pred[:,0]) 
    loss_pinn = metric(y[:,0],y_pred[:,2]) 
    losses = [loss_spvsd, loss_pinn]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

def exp_grad_hess_loss(func,x,y, loss_weights, metric):
    y_pred = func(x)
    loss_spvsd = metric(y[:,0],y_pred[:,0]) 
    loss_pinn1 = metric(y[:,0],y_pred[:,1]) 
    loss_pinn2 = metric(y[:,0],y_pred[:,2]) 
    losses = [loss_spvsd, loss_pinn1, loss_pinn2]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

# Trigonometric
def trig_hess_loss(func,x,y, loss_weights, metric):
    y_pred = func(x)
    loss_spvsd = metric(y[:,0],y_pred[:,0]) 
    loss_pinn = metric(y[:,0],-y_pred[:,2]) 
    losses = [loss_spvsd, loss_pinn]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

# BS
def BS_SPINN(func,x,y,loss_weights,metric,r,sigma):
    y_pred = func(x)
    loss_spvsd = metric(r*y[:,0],r*y_pred[:,0]) 
    equation = y_pred[:,1]-0.5*sigma*sigma*x[:,1]*x[:,1]*y_pred[:,-1]*y_pred[:,-1]-r*x[:,1]*y_pred[:,2]+r*y[:,0]
    loss_pinn = metric(0.,equation) 
    losses = [loss_spvsd, loss_pinn]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss

