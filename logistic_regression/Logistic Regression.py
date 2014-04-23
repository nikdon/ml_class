
# coding: utf-8

# # Machine Learning Online Class - Exercise 2 Logistic Regression
# 
# Start the exercise by first loading and plotting the data to understand the problem:

# In[1]:

import numpy as np


loaded_data = np.loadtxt('ex2data1.txt', delimiter=',')
X = loaded_data[:,:2]
y = loaded_data[:,-1]

print X[0:5, :]
print y[0:5]


# In[2]:

from matplotlib import pyplot as plt

pos = y > 0
neg = y == 0

plt.scatter(X[pos,0], X[pos,1], marker='+', color='r', s=35)
plt.scatter(X[neg,0], X[neg,1], marker='o', color='b', s=20)
plt.show()


# Define sigmoid function $g(z) = \frac{1}{1 - e^z}$ and sigmoid gradient function $g'(z)=\frac{d}{dz}g(z) = g(z)(1-g(z))$:

# In[3]:

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

x = np.arange(-6, 6, 0.1)

subplot(2,1,1)
plt.plot(x, sigmoid(x))
plt.grid(True)

subplot(2,1,2)
plt.plot(x, sigmoid_gradient(x))
plt.grid(True)

plt.show()


# Write function computes the cost $J(\theta)$ and gradient $\frac{d}{d\theta}J(\theta)$:

# In[14]:

def cost_function(theta, X, y, lam=0):
    m = X.shape[0]

    predictions = sigmoid(X * c_[theta])
    J = 1./m * (-y.T.dot(np.log(predictions)) - (1-y).T.dot(np.log(1 - predictions)))
    
    # Add regularization
    J += lam/(2.0*m) * np.sum(np.dot(c_[theta].T, c_[theta]))
    
    grad = 1./m * X.T * (predictions - y) + lam * c_[theta]
    
    return J, grad


# ## Part 2: Compute Cost and Gradient

# In[15]:

loaded_data = np.loadtxt('D:\SkyDrive\Courses\ML Class\mlclass-ex2-005\mlclass-ex2\ex2data1.txt', delimiter=',')
X = loaded_data[:,:2]
y = loaded_data[:,-1]

m, n = X.shape

X = mat(c_[loaded_data[:, :2]])
y = c_[loaded_data[:, 2]]

X = c_[np.ones(m), X]

initial_theta = np.zeros((n+1, 1))

cost, grad = cost_function(initial_theta, X, y, lam=0)

print 'Cost: ', cost
print 'Gradient: ', grad.T


# ## Part 3: Optimizing
# 

# In[17]:

from scipy import optimize

res = optimize.minimize(lambda t: cost_function(t, X, y, lam=0)[0],
                                  initial_theta, 
                                  method='Nelder-Mead',
                                  options={'disp':True, 'disp':True},
                                  jac=False)

print res


# ## Part 4: Predict Accuracies

# In[18]:

def predict(theta, X):
    p = sigmoid(X * c_[theta]) >= 0.5
    return p


nn_params = res.x
p = predict(nn_params.T, X)

print 'Training Set Accuracy: %f' % (np.mean(p == y) * 100)

