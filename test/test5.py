import tensorflow as tf

def calc_hessian_diag(f, x):
    """
    Calculates the diagonal entries of the Hessian of the function f
    (which maps rank-1 tensors to scalars) at coordinates x (rank-1
    tensors).
    
    Let k be the number of points in x, and n be the dimensionality of
    each point. For each point k, the function returns

      (d^2f/dx_1^2, d^2f/dx_2^2, ..., d^2f/dx_n^2) .

    Inputs:
      f (function): Takes a shape-(k,n) tensor and outputs a
          shape-(k,) tensor.
      x (tf.Tensor): The points at which to evaluate the Laplacian
          of f. Shape = (k,n).
    
    Outputs:
      A tensor containing the diagonal entries of the Hessian of f at
      points x. Shape = (k,n).
    """
    # Use the unstacking and re-stacking trick, which comes
    # from https://github.com/xuzhiqin1990/laplacian/
    with tf.GradientTape(persistent=True) as g1:
        # Turn x into a list of n tensors of shape (k,)
        #x_unstacked = tf.unstack(x, axis=1)
        g1.watch(x)

        with tf.GradientTape() as g2:
            # Re-stack x before passing it into f
            #x_stacked = tf.stack(x_unstacked, axis=1) # shape = (k,n)
            g2.watch(x)
            f_x = f(x) # shape = (k,)
        
        # Calculate gradient of f with respect to x
        df_dx = g2.gradient(f_x, x) # shape = (k,n)
        # Turn df/dx into a list of n tensors of shape (k,)
        df_dx_unstacked = tf.unstack(df_dx, axis=1)

    # Calculate 2nd derivatives
    d2f_dx2 = []
    for df_dxi in df_dx_unstacked:
        # Take 2nd derivative of each dimension separately:
        #   d/dx_i (df/dx_i)
        d2f_dx2.append(g1.gradient(df_dxi, x))
    
    # Stack 2nd derivates
    d2f_dx2_stacked = tf.stack(d2f_dx2, axis=1) # shape = (k,n)
    
    return d2f_dx2_stacked
f = lambda q : tf.math.log(tf.math.reduce_sum(q**2, axis=1))
x = tf.random.uniform((5,3))

d2f_dx2 = calc_hessian_diag(f, x)
print(d2f_dx2)
