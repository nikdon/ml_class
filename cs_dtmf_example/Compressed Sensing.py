# coding: utf-8

# http://brocabrain.blogspot.ru/2012/10/compressed-sensing-with-sklearn-dtmf.html
# 
# ## Magic Reconstruction
# 
# I first read about compressed sensing about two years ago, on Cleve's Corner, a column which Cleve Moler, the founder of Mathworks, writes anually. The 2010 article was called "Magic Reconstruction: Compressed Sensing". Compressed (or compressive) sensing is immediately appealing for a number of reasons. It has all the features of a cool signal processing problem. It seems to beat the Shannon-Nyquist sampling theorem by making use of sparsity in real signals, and this sparsity begs the question - sparsity in which basis? This question in turn calls for some prior information about the data in question, which potentially leads to a wide range of other mathematical problems. (Perhaps a machine learning problem for detecting which basis are appropriate for a given set of signals, especially when prior information about signals is not available.)
# 
# But even without all this, as Moler points out, the real appeal of compressed sensing lies in the underlying matrix problem. This problem requires a great deal of rigour in linear algebra and convex optimization. I did get to those problems eventually, but initially I was more curious as to whether compressed sensing really worked, and I tried out the example in his post in Python. That's what this post is about.
# 
# ## Python Implementation of the DTMF Example
# 
# The problem consists simply of creating a touchtone signal, trying to compress and decompress it, and checking whether the original signal and the reconstructed signal match acoustically. Let's step through the programme step by step.

# In[1]:

# Imports
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
from matplotlib.pyplot import plot, show, figure, title
import numpy as np


# We pick any two touchtone frequencies $f_1$ and $f_2$ (in this case 697 Hz and 1336 Hz, corresponding to the '5' key on the keypad), and play the following signal for an eighth of a second.
# $f = \sin(2\pi f_1t) + \sin(2\pi f_1t)$
# At a sampling rate of 4 kHz, it comes to 5000 samples. As per the example, we take 500 random samples of this signal.

# In[11]:

# Initializing constants and signals
N = 5000
FS = 4e4
M = 500
f1, f2 = 697, 1336  # Pick any two touchtone frequencies
duration = 1. / 8
t = np.linspace(0, duration, duration * FS)
f = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
f = np.reshape(f, (len(f), 1))

# Displaying the test signal
plot(t, f)
title('Original Signal')
show()

# Randomly sampling the test signal
k = np.random.randint(0, N, (M,))
k = np.sort(k)  # making sure the random samples are monotonic
b = f[k]
plot(t, f, 'b', t[k], b, 'r.')
title('Original Signal with Random Samples')
show()


# Since this is a simple, almost stationary signal, a simple basis like discrete cosines should suffice to bring out the sparsity.

# In[12]:

D = np.fft.fft(np.eye(N))
A = D[k, :]


# Here $A$ is a matrix which contains a subset of 500 discrete cosine bases, and we need to solve $Ax=b$ for $x$. It is a nonlinear optimization problem and there are many solutions, but it turns out that the one that minimizes the $L_1$ norm of the solution gives the best estimate of the original signal. Since this is an optimization problem, it can be solved with many of the methods in scipy.optimize, say by taking the least squares solution of the equation (or the $L_2$ norm) as the first guess and minimizing iteratively. But I took the easier approach and used the Lasso estimator in the sklearn package, which is essentially a linear estimator that penalizes (regularizes) its weights in the $L_1$ sense. (A really cool demonstration of compressed sensing for images using Lasso is [here](http://scikit-learn.org/0.14/auto_examples/applications/plot_tomography_l1_reconstruction.html)).

# In[18]:

lasso = Lasso(alpha=0.001)
lasso.fit(A, b.reshape((M,)))

# Plotting the reconstructed coefficients and the signal
plot(lasso.coef_)
# xlim([0, 500])
title('FFT of the Reconstructed Signal')
recons = np.fft.ifft(lasso.coef_.reshape((N, 1)), axis=0)
figure()
plot(t, recons)
title('Reconstucted Signal')
show()


# As can be seen through the plots, most of the coefficients of the lasso estimator as zeros. It is the discrete cosine transform of these coefficients that is the reconstructed signal. Since the coefficients are sparse, they can be compressed into a scipy.sparse matrix.

# In[8]:

recons_sparse = coo_matrix(lasso.coef_)
sparsity = 1 - float(recons_sparse.getnnz()) / len(lasso.coef_)
print sparsity


# As it turns out, the compressed matrix is about 90% sparse. Thus, we have managed to reconstruct the signal from only 10% of its samples.
# 
# ## Validating the Reconstruction
# 
# A reasonably reliable method of validating the compression and reconstruction is to listen to the original  signal and check if the reconstructed signal sounds similar. The scikits.audiolab package can be used to play sound straight from numpy arrays. (I couldn't make audiolab work, so I validated this by saving the reconstructed array into a .mat file and playing it out in Octave.) The python codes used here are available as an ipython notebook and a python script in this repository.
# 
# ## Further...
# 
# The entire reconstruction depends on sparsity and prior information about the signal. That's probably why wavelets are a popular choice for bases in compressed sensing applications, since using wavelets is equivalent to imposing a prior basis on signals. The important question to answer in all compressed sensing problems is - in which basis is the signal sparse? An interesting follow up question would be - what if the bases are intrinsic mode functions? How does one perform empirical mode decomposition such that the decomposition is sparse? And that's only the tip of the iceberg. Prior information is exactly what is absent in intrinsic mode functions! Examining IMFs as bases for sparsity in signals should be an interesting problems, given that they complement wavelets nicely.
