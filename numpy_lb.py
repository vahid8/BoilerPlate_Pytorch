# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:28:47 2020

@author: vaghajan
"""

import numpy as np
import matplotlib.pyplot as plt
'''
# -----------------------------------------------> Create np
a= np.array (range(9),float).reshape (3,3)

b = np.arange(5, dtype=float)
c = np.arange(1, 6, 2, dtype=int)

d = np.ones((2,3), dtype=float)
e = np.zeros(7, dtype=int)

f = np.identity(4, dtype=float)
g = np.eye(4, k=1, dtype=float) #The eye function returns matrices with ones along the kth diagonal

print(np.pi)
print(np.e)
 
# -----------------------------------------------> Transpose matrix
a = a.transpose()
# -----------------------------------------------> Bring all into one line
a = a.flatten()

# -----------------------------------------------> Concatenate 
a = np.array([[1, 2], [3, 4]], float) 
b = np.array([[5, 6], [7,8]], float) 
c = np.concatenate((a,b))
print(c) 
c = np.concatenate((a,b), axis=0) 
print(c)
c = np.concatenate((a,b), axis=1) 
print(c)


# -----------------------------------------------> dimension addition
#a = np.array([1,2,3],float)
#print(a.shape)
#a = a[:,np.newaxis]
#print(a.shape)


# -----------------------------------------------> Copy with zeros
#The zeros_like and ones_like functions create a new array
# with the same dimensions and type of an existing one
a = np.array([[1, 2, 3], [4, 5, 6]], float)
b=  np.zeros_like(a) #np.ones_like(a)
print(b)

# -----------------------------------------------> round numbers
#The functions floor, ceil, and rint give the lower, upper, or nearest (rounded) integer:

# -----------------------------------------------> Array iteration
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for x in a:
    print(a)
for (x,y) in a:
    print(x*y)

# -----------------------------------------------> Basic array operations
a = np.array([2, 4, 3], float)
# sum of all members
a.sum()
np.sum(a)

# Multipication of all members
a.prod()
np.prod(a)

a.mean()
a.var()
a.std()
a.min()
a.max()

# -----------------------------------------------> find max and min indices
a.argmin()
a.argmax()

# -----------------------------------------------> calculation along rows or columns
a = np.array([[0, 2], [3, -1], [3, 5]], float)
a.mean(axis=0) #output = array([ 2., 2.]) along cols
a.mean(axis=1) #output = array([ 1., 1., 4.]) along rows

# -----------------------------------------------> sorting or limitimg array
a = np.array([6, 2, 5, -1, 0], float)
a.sort() # array([-1., 0., 2., 5., 6.])
a.clip(0, 5) # array([ 5., 2., 5., 0., 0.])

# -----------------------------------------------> extract unique values
np.unique(a)

# -----------------------------------------------> extract diage
a.diagonal()

# -----------------------------------------------> Comparison operators and value testing
a = np.array([1, 3, 0], float)
b = np.array([0, 3, 2], float)
a > b # output = array([ True, False, False], dtype=bool)
a > 2 # output =  array([False, True, False], dtype=bool)

#The any and all operators can be used to determine
# whether or not any or all elements of a Boolean array are true
c = np.array([ True, False, False], bool)
any(c) #output: True
all(c) # output : False

#Compound Boolean expressions can be applied to arrays
# on an element-by-element basis using special functions
# logical_and, logical_or, and logical_not.
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3) #output: array([ True, False, False], dtype=bool)
b = np.array([True, False, True], bool)
np.logical_not(b) #output : array([False, True, False], dtype=bool)
c = np.array([False, True, False], bool)
np.logical_or(b, c) #output: array([ True, True, True], dtype=bool)

#The where function forms a new array from two arrays of equivalent
# size using a Boolean filter to choose between elements of the two.
# Its basic syntax is where(boolarray, truearray, falsearray)
a = np.array([1, 3, 0], float)
np.where(a != 0, 1 / a, a) #output : array([ 1. , 0.33333333, 0. ])

#Broadcasting can also be used with the where function:
np.where(a > 0, 3, 2) # output : array([3, 3, 2])

#A number of functions allow testing of the values in an array.
# The nonzero function gives a tuple of indices of the nonzero values in an array.
# The number of items in the tuple equals the number of axes of the array:
a = np.array([[0, 1], [3, 0]], float)
a.nonzero() #output : (array([0, 1]), array([1, 0]))

# -----------------------------------------------> chack NAN and Inf
a = np.array([1, np.NaN, np.Inf], float)
np.isnan(a)  #array([False, True, False], dtype=bool)
np.isfinite(a) # array([ True, False, False], dtype=bool)

# -----------------------------------------------> Array item selection and manipulation
a = np.array([[6, 4], [5, 9]], float)
a >= 6 #output array([[ True, False], [False, True]], dtype=bool)
a[a >= 6] #output array([ 6., 9.])

# -----------------------------------------------> Vector and matrix mathematics
# dot product or inner product
np.dot(a, b)
np.inner(a, b)

# outer product
np.outer(a, b)

#determinan
np.linalg.det(a)

# eigenvalues and eigenvectors of a matrix
vals, vecs = np.linalg.eig(a)

# The inverse matrix
b = np.linalg.inv(a)

# Singular value decomposition
U, s, Vh = np.linalg.svd(a)

# -----------------------------------------------> Polynomial mathematics
# polynomial definition
#1. given roots and get the polynomial coefficients
np.poly([-1, 1, 1, 10]) #output : array([ 1, -11, 9, 11, -10])  -> 𝑥4−11𝑥3+9𝑥2+11𝑥−10
#2. given set of coefficients and get the polynomial roots
np.roots([1, 4, -2, 3]) #output array([-4.57974010+0.j , 0.28987005+0.75566815j, 0.28987005-0.75566815j])

#integral eg 𝑥3+𝑥2+𝑥+1
np.polyint([1, 1, 1, 1]) #output : array([ 0.25 , 0.33333333, 0.5 , 1. , 0. ]) -> 𝑥4/4+𝑥3/3+𝑥2/2+𝑥+𝐶

# derivative eg 𝑥3+𝑥2+𝑥+1
np.polyder([1./4., 1./3., 1./2., 1., 0.])

#evaluates a polynomial at a particular point
np.polyval([1, -2, 0, 2], 4)

#fit a polynomial of specified order to a set of data using a least-squares approach
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 2, 1, 3, 7, 10, 11, 19]
np.polyfit(x, y, 2)
#output array([ 0.375 , -0.88690476, 1.05357143])
#More sophisticated interpolation routines can be found in the SciPy package



# -----------------------------------------------> Statistics
# median
a = np.array([1, 4, 3, 8, 9, 2, 3], float)
np.median(a)

#Correlation coefficients
#for arrays of the form [[x1, x2, …], [y1, y2, …], [z1, z2, …], …] where x, y, z are different
#observables and the numbers indicate the observation times:
a = np.array([[1, 2, 1, 3], [5, 3, 1, 8]], float)
c = np.corrcoef(a)

# Covariance of data
np.cov(a)

# -----------------------------------------------> Random numbers generation
np.random.rand(5)
# To generate random integers in the range [min, max)
np.random.randint(5, 10)

# To draw from the discrete Poisson distribution with 𝜆 = 6.0,
np.random.poisson(6.0)

# To draw from a continuous normal (Gaussian) distribution with mean 𝜇 = 1.5 and standard
# deviation 𝜎 = 4.0
np.random.normal(1.5, 4.0)

#To draw from a standard normal distribution (𝜇 = 0, 𝜎 = 1), omit the arguments
np.random.normal()

#To draw multiple values, use the optional size argument:
np.random.normal(size=5)

#The random module can also be used to randomly shuffle the order of items in a list. This is
#sometimes useful if we want to sort a list in random order
l = range(10)
np.random.shuffle(l)

#The random number seed can be set
np.random.seed(293423)

