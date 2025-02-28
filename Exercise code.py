#Exercise code


# exercise 3.1.1
import importlib_resources
import numpy as np
import xlrd

# Load xls sheet with data
filename = importlib_resources.files("dtuimldmtools").joinpath("data/nanonose.xls")
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 3, 11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0, 2, 92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((90, 8))
for i, col_id in enumerate(range(3, 11)):
    X[:, i] = np.asarray(doc.col_values(col_id, 2, 92))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print("Ran Exercise 3.1.1")

# exercise 3.1.3
# (requires data structures from ex. 3.1.1)
import matplotlib.pyplot as plt
from ex3_1_1 import *
from scipy.linalg import svd 

# Subtract mean value from data
# Note: Here we use Y to in teh book we often use X with a hat-symbol on top.
Y = X - np.ones((N, 1)) * X.mean(axis=0)

# PCA by computing SVD of Y
# Note: Here we call the Sigma matrix in the SVD S for notational convinience
U, S, Vh = svd(Y, full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components 
# Note: This is an important equation, see Eq. 3.18 on page 40 in the book.
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

print("Ran Exercise 3.1.3")













# exercise 3.1.3
# (requires data structures from ex. 3.1.1)
import matplotlib.pyplot as plt
from ex3_1_1 import *
from scipy.linalg import svd 

# Subtract mean value from data
# Note: Here we use Y to in teh book we often use X with a hat-symbol on top.
Y = X - np.ones((N, 1)) * X.mean(axis=0)

# PCA by computing SVD of Y
# Note: Here we call the Sigma matrix in the SVD S for notational convinience
U, S, Vh = svd(Y, full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components 
# Note: This is an important equation, see Eq. 3.18 on page 40 in the book.
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

print("Ran Exercise 3.1.3")


# SVD


##What we need to have for PCA, 

# we need to have the explained variance for principal components, se exercise ex3_1_3, 
# so we know how many og the principal components explain the majority of the variance. 

#Need to do pca of 1 and second principal component. 
'