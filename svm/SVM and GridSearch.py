
# coding: utf-8

# # Support Vector Machines
# 
# ## Part 1: Loading and Visualizing Data
# 
# We start the exercise by first loading and visualizing the dataset. The following code will load the dataset into your environment and plot the data:

# In[1]:

import numpy as np
import scipy.io

ex6data2 = scipy.io.loadmat('ex6data2.mat')

X = ex6data2['X']
y = ex6data2['y']

print 'X: ', X[0:5, :]
print 'y: ', y[0:5].ravel()


# In[2]:

from matplotlib import pyplot as plt

pos = np.where( y == 1.0 )
neg = np.where( y == 0.0 )

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.show()


# ## Part 2: Training SVM with radial basis function kernel
# 
# Note about Kernel functions: the *kernel function* can be any of the following:
# - linear: $\langle{x, x'}\rangle$
# - polinomial: $(\gamma\langle{x, x'}\rangle + r)^d$, **d** is specified by keyword **degree**, **r** by **coef0**.
# - rbf: $\langle{-\gamma|x - x'|^2}\rangle$, $\gamma$ is specified by keyword **gamma**, must be greater than 0
# - sigmoid: $tanh(-\gamma\langle{x, x'}\rangle + r)$, where r is specified by coef0
# 
# The following code will train a SVM on the dataset:

# In[3]:

from sklearn import svm

clf = svm.SVC(C=1, gamma=1/(2 * 0.1 **2), kernel='rbf')
clf.fit(X, y.ravel())


# and plot the decision boundary learned:

# In[4]:

# step size in the mesh
h = .01

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.show()


# ## Part 3: Training SVM with radial basis function kernel using cross validation set
# 
# This code snippet returns best choice of **C** and **sigma** for SVM with RBF kernel:

# In[9]:

# Train classifier
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

from sklearn.grid_search import GridSearchCV


ex6data3 = scipy.io.loadmat('ex6data3.mat')
X = ex6data3['Xval']
y = ex6data3['yval']
Xval = ex6data3['Xval']
yval = ex6data3['yval']

C_range = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
gamma_range = np.array([1/(2 * 0.01 **2), 
                        1/(2 * 0.03 **2), 
                        1/(2 * 0.1 **2), 
                        1/(2 * 0.3 **2), 
                        1/(2 * 1 **2), 
                        1/(2 * 3 **2), 
                        1/(2 * 10 **2), 
                        1/(2 * 30 **2)])
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(C=1.0), param_grid=param_grid)
grid.fit(Xval, yval.ravel())

print("The best classifier is: ", grid.best_estimator_)


# step size in the mesh
h = .02

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = grid.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.show()

