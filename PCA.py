# PCA analysis of the data 


#Prior to the PCA analysis, all of the attributes should be standadised, 
# and investigated for any outlisers, if there is outliers,
#  log-transformations could 
#be applied to thosede attributes. 

#imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# read the data and create a dataframe using pandas
data = pd.read_csv('cleaned_cleveland.csv')


#Saving the amount of observations
N = len(data)

#standardizing the data (or normalizing it?)
# Subtract mean collumn value from each element in each collumn
stand_data=data.sub(data.mean(axis=0), axis=1)
print("N",N)
print("stand_data", stand_data)

mean_stand_data = stand_data.mean(axis=0)
print("mean_stand_data", mean_stand_data)

#PCA by computing SVD of stand_data

U, S, Vh = svd(stand_data, full_matrices=False)

#Get V from Vh = V.hermitian
V = Vh.T

# Compute variance explained by principal components - INPORTANT EQUATION
rho = (S * S) / (S * S).sum()

#apply standard threshold
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
plt.savefig('Variance_explained_by_principal_components.png')


#******************************************



#PCA of first and second principal component 


# Project the centered data onto principal component space
# Note: Make absolutely sure you understand what the @ symbol 
# does by inspecing the numpy documentation!
Z = stand_data @ V

# Indices of the principal components to be plotted
i = 0
j = 1


# Plot PCA of the data
f = plt.figure()
plt.title("NanoNose data: PCA")
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
#plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
plt.show()

