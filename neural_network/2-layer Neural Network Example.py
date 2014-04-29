
# coding: utf-8

# # Machine Learning Online Class - Exercise 4 Neural Network Learning
# 
# Setup the parameters:

# In[117]:

import numpy as np

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10


# ##Part 1: Loadind and Visualizing Data
# 
# So we start the exercise by first loading and visualizing the dataset. Dataset contains handwritten digits.
# 
# Load training data:

# In[118]:

import scipy.io

ex4data1 = scipy.io.loadmat('ex4data1.mat')

X = ex4data1['X']
y = ex4data1['y']


# And randomly select sample to display:

# In[119]:

sel = np.random.randint(X.shape[0])
    
from matplotlib import pyplot as plt

plt.imshow(np.rot90(X[sel,:].reshape(20, 20)), origin='lower', cmap = cm.Greys_r)
plt.show()


# ## Part 2: Loading parameters
# 
# Now we have to load and pre-initialize neural network paramteres $\theta_1$ and $\theta_2$:

# In[120]:

ex4weights = scipy.io.loadmat('ex4weights.mat')

Theta1 = ex4weights['Theta1']
Theta2 = ex4weights['Theta2']


# ## Part 3: Compute Cost (Feedforward)
# 
# To the neural network, first we should start by implementing the feedforward part of the neural network that returns the cost only. We have to write the code to return cost. After implementing the feedforward to compute the cost, we can verify implementation is correct by verifying that we get the same cost as us for the fixed debugging parameters. 
# 
# We suggest implementing the feedforward cost *without* regularization first so that it will be easier to debug. Later, in part 4, you will get to implement the regularized cost.
# 
# So now set weitght regularization parameter $\lambda$ to 0:

# In[121]:

wrp_lambda = 0


# And define sigmoid function $g(z) = \frac{1}{1 - e^z}$ and sigmoid gradient function $g'(z)=\frac{d}{dz}g(z) = g(z)(1-g(z))$:

# In[122]:

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

x = np.arange(-6, 6, 0.1)

subplot(2,1,1)
plt.plot(x, sigmoid(x))
plt.grid(True)

subplot(2,1,2)
plt.plot(x, sigmoid_gradient(x))
plt.grid(True)

plt.show()


# And write function computes the cost $J(\theta)$ and gradient $\frac{d}{d\theta}J(\theta)$ of the neural network, where **nn_params** is tuple $(\theta^{(1)}$, $\theta^{(2)})$:

# In[123]:

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, wrp_lambda=0):
    
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))
    
    m = X.shape[0]
    
    J = 0.0
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    a1 = np.vstack((np.ones(m), X.T))
    z2 = np.dot(Theta1, a1)
    a2 = np.vstack((np.ones(m), sigmoid(z2)))
    a3 = sigmoid(np.dot(Theta2, a2))
    
    Y = np.zeros([Theta2.shape[0], m]);
    
    for col in xrange(np.size(y)):
        Y[y[col] - 1, col] = 1.0
    
    J = (1.0/m) * np.sum(np.sum(-Y * np.log(a3) - (1.0 - Y) * np.log(1.0 - a3)));
    
    # Add regularization
    J += wrp_lambda/(2.0*m) * (np.sum(Theta1**2) + np.sum(Theta2**2))
    
    # Implement the backpropagation algorithm to compute the gradients
    d3 = a3 - Y
    d2 = (np.dot(Theta2.T, d3)) * np.vstack((np.ones(m), sigmoid_gradient(z2)))
    
    Theta2_grad = 1.0/m * np.dot(d3, a2.T)
    Theta1_grad = 1.0/m * np.dot(d2[1:, :], a1.T)
    
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    
    return J, grad


nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, wrp_lambda=0.0)
print 'Cost at parameters (loaded from ex4weights): %f' % J

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, wrp_lambda=1.0)
print 'Cost at parameters (loaded from ex4weights) w/ regularization: %f' % J


# ## Part 4: Initializing Parameters and Training
# 
# Now we start to implement a two layer network classifies digits. First we have to write a function that initialize the weights of neural network:

# In[124]:

def rand_init_weight(layers_in, layers_out, eps_init=0.12):
    weigths = np.random.randn(layers_in + 1, layers_out) * 2 * eps_init - eps_init
    return weigths.T


initial_Theta1 = rand_init_weight(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_init_weight(hidden_layer_size, num_labels)


# We almost done with all the code necessary to train neural network. For that purpose we will use [scipy.optimize module](/http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize/). We have to remember that our cost funstion returns $J(\theta)$ and $\frac{d}{d\theta}J(\theta)$, so we have to set optional **jac** parameter to **True**. For additional information one may check discussion [here](/http://stackoverflow.com/questions/17431070/how-to-return-cost-grad-as-tuple-for-scipys-fmin-cg-function/).

# In[125]:

from scipy import optimize

wrp_lambda = 1.0
nn_params = np.hstack((initial_Theta1.flatten(), initial_Theta2.flatten()))
args = (input_layer_size, hidden_layer_size, num_labels, X, y, wrp_lambda)

res = optimize.minimize(nn_cost_function, nn_params, args=args, method='CG', options={'maxiter':100,'disp':True}, jac=True)


# The optimization result represented as a *Result* object. Important attributes is **x** - the solution array. When it has been obrained, we can reshape it to $\theta^{(1)}$ and $\theta^{(2)}$, which represent optimized weight parameters:

# In[126]:

nn_params = res.x

Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))


# After training the neural network, we would like to use it to predict the labels. We implement the "predict" function to use the neural network to predict the labels of the training set. This lets us compute the training set accuracy:

# In[127]:

def predict(theta1, theta2, X):

    m = X.shape[0]
    h1 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), X)), Theta1.T))
    h2 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), h1)), Theta2.T))

    p = np.argmax(h2, axis=1)

    return p


p = predict(Theta1, Theta2, X)
print 'Training Set Accuracy: %f' % (np.mean(p == y.flatten() - 1) * 100)

