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

def exp_loss(func,x,y, loss_weights, metric):
    y_pred = func(x)
    loss_spvsd = metric(y[:,0],y_pred[:,0]) 
    loss_pinn = metric(y[:,0],y_pred[:,1]) 
    losses = [loss_spvsd, loss_pinn]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    losses *= loss_weights
    total_loss = tf.math.reduce_sum(losses)
    return total_loss
