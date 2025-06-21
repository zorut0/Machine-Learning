#a

import math

x = 5.7
print("#a\nFloor:", math.floor(x))      
print("Ceil:", math.ceil(x))         
print("Square root:", math.sqrt(16)) 
print("Integer sqrt:", math.isqrt(17)) 
print("GCD:", math.gcd(36, 60))      

#b

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("\n#b\nDimensions:", arr.ndim)       
print("Shape:", arr.shape)           
print("Size:", arr.size)             
print("Sum:", arr.sum())             
print("Mean:", arr.mean())           
print("Sorted array:\n", np.sort(arr, axis=1))
print("Sine values:\n", np.sin(arr))
print("Cose values:\n", np.cos(arr))
print("Tan values:\n", np.tan(arr))

#c

from numpy.linalg import det, eig

matrix = np.array([[4, 2], [3, 1]])

print("\n#c\nDeterminant:", det(matrix))  

eigenvalues, eigenvectors = eig(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

#d


lst = list(range(12))
arr = np.array(lst)

arr_2d = arr.reshape(3, 4)
print("\n#d\n2D Array:\n", arr_2d)

arr_3d = arr.reshape(2, 2, 3)
print("3D Array:\n", arr_3d)


#e

print("\n#e\nZero matrix:\n", np.zeros((3, 3)))

print("Ones matrix:\n", np.ones((2, 4)))

print("Identity matrix:\n", np.eye(4))


#f


from scipy.linalg import det as scipy_det

matrix = np.array([[1, 2], [3, 4]])
print("\n#f\nDeterminant using SciPy:", scipy_det(matrix))

#g

from scipy.linalg import eig as scipy_eig

matrix = np.array([[5, 2], [2, 1]])
eigenvalues, eigenvectors = scipy_eig(matrix)

print("\n#g\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)






