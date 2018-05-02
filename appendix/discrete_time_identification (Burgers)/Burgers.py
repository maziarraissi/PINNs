"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q):
        
        self.lb = lb
        self.ub = ub
        
        self.x0 = x0
        self.x1 = x1
        
        self.u0 = u0
        self.u1 = u1
        
        self.layers = layers
        self.dt = dt
        self.q = max(q,1)
    
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)       
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
        self.IRK_alpha = weights[0:-1,:]
        self.IRK_beta = weights[-1:,:]        
        self.IRK_times = tmp[q**2+q:]
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, self.x0.shape[1]))
        self.x1_tf = tf.placeholder(tf.float32, shape=(None, self.x1.shape[1]))
        self.u0_tf = tf.placeholder(tf.float32, shape=(None, self.u0.shape[1]))
        self.u1_tf = tf.placeholder(tf.float32, shape=(None, self.u1.shape[1]))
        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q)) # dummy variable for fwd_gradients        
        self.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, self.q)) # dummy variable for fwd_gradients        
        
        self.U0_pred = self.net_U0(self.x0_tf) # N0 x q
        self.U1_pred = self.net_U1(self.x1_tf) # N1 x q
        
        self.loss = tf.reduce_sum(tf.square(self.u0_tf - self.U0_pred)) + \
                    tf.reduce_sum(tf.square(self.u1_tf - self.U1_pred)) 
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def fwd_gradients_0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]
    
    def fwd_gradients_1(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x1_tf)[0]
        return tf.gradients(g, self.dummy_x1_tf)[0]    
    
    def net_U0(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        U = self.neural_net(x, self.weights, self.biases)        
        U_x = self.fwd_gradients_0(U, x)
        U_xx = self.fwd_gradients_0(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U0 = U - self.dt*tf.matmul(F, self.IRK_alpha.T)
        return U0
    
    def net_U1(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        U = self.neural_net(x, self.weights, self.biases)        
        U_x = self.fwd_gradients_1(U, x)
        U_xx = self.fwd_gradients_1(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U1 = U + self.dt*tf.matmul(F, (self.IRK_beta - self.IRK_alpha).T)
        return U1

    def callback(self, loss):
        print('Loss:', loss)
    
    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0, 
                   self.x1_tf: self.x1, self.u1_tf: self.u1,
                   self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),
                   self.dummy_x1_tf: np.ones((self.x1.shape[0], self.q))}
                           
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = np.exp(self.sess.run(self.lambda_2))
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
    
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
    
    def predict(self, x_star):
        
        U0_star = self.sess.run(self.U0_pred, {self.x0_tf: x_star, self.dummy_x0_tf: np.ones((x_star.shape[0], self.q))})        
        U1_star = self.sess.run(self.U1_pred, {self.x1_tf: x_star, self.dummy_x1_tf: np.ones((x_star.shape[0], self.q))})
                    
        return U0_star, U1_star

    
if __name__ == "__main__": 
        
    skip = 80

    N0 = 199
    N1 = 201
    
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t_star = data['t'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    Exact = np.real(data['usol'])
    
    idx_t = 10
        
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0    
    
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:]
    u0 = Exact[idx_x,idx_t][:,None]
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
        
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:]
    u1 = Exact[idx_x,idx_t + skip][:,None]
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    dt = np.asscalar(t_star[idx_t+skip] - t_star[idx_t])        
    q = int(np.ceil(0.5*np.log(np.finfo(float).eps)/np.log(dt)))
    
    layers = [1, 50, 50, 50, 50, q]

    
    # Doman bounds
    lb = x_star.min(0)
    ub = x_star.max(0)

    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)
    model.train(nIter = 50000)
    
    U0_pred, U1_pred = model.predict(x_star)    
        
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = np.exp(model.sess.run(model.lambda_2))
                
    nu = 0.01/np.pi       
    error_lambda_1 = np.abs(lambda_1_value - 1.0)/1.0 *100
    error_lambda_2 = np.abs(lambda_2_value - nu)/nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1))
    print('Error lambda_2: %f%%' % (error_lambda_2))
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01        
    
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)    
    model.train(nIter = 50000)
    
    U_pred = model.predict(x_star)
    
    U0_pred, U1_pred = model.predict(x_star)    
        
    lambda_1_value_noisy = model.sess.run(model.lambda_1)
    lambda_2_value_noisy = np.exp(model.sess.run(model.lambda_2))
         
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)/1.0 *100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu)/nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    fig, ax = newfig(1.0, 1.5)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
        
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',
                  extent=[t_star.min(),t_star.max(), lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]
    ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
    ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)
    
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)
    
    ax = plt.subplot(gs2[0, 0])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + %.6f u_{xx} = 0$ \\  \hline Identified PDE (clean data) & ' % (nu)
    s2 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(-0.1,0.2,s)

    # savefig('./figures/Burgers') 