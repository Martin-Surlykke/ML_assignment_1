# PCA analysis of the data 

### NOTE PCA looks a little wierd so look into it more, - also consider doing log tranformations. 

#Prior to the PCA analysis, all of the attributes should be standadised, 
# and investigated for any outlisers, if there is outliers,
#  log-transformations could 
#be applied to thosede attributes. 

#imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
<<<<<<< Updated upstream
from Correlation_matrix import * #extract_relevant_vals



=======
import Correlation_matrix as cm
>>>>>>> Stashed changes

# read the data and create a dataframe using pandas
data = pd.read_csv('cleaned_cleveland.csv')

relevant_vals=extract_relevant_vals(data)
print(relevant_vals.head())
data = relevant_vals

#Saving the amount of observations
N = len(data)

#Saving the "num" as the class label
<<<<<<< Updated upstream
y = data["num"] 
#Saving the diffirent categories of the class label
=======
y= data["num"].apply(lambda x: 1 if x > 0 else 0)

y_binary = y.copy()

for row in y_binary:
    if row > 0:
        y_binary = y_binary.replace(row, 1)

#For PCA we dont include the num attribute
stand_data = data.drop(columns = ["num"])
#Savinf the diffirent categories of the class label
>>>>>>> Stashed changes
C = len(y.unique())  


print("C",C)
print("y",y)
print("y done")

#First we apply log transformation 
data["ca"] = np.log(data["ca"]+1)
data["oldpeak"] = np.log(data["oldpeak"]+1)
print("log tranformation applied on Ca and Oldpeak")
#standardizing the data (or normalizing it?)
# Subtract mean collumn value from each element in each collumn
stand_data= cm.normalize_data(data)


print("stand_describe")
print(stand_data.describe())

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
plt.savefig('images/Variance_explained_by_principal_components.png')
 
#PCA of first and second principal component 
# Project the centered data onto principal component space
Z = stand_data @ V

# Indices of the principal components to be plotted
i = 0
j = 1


# Plot PCA of the data
f = plt.figure()
plt.title("Heart Disease data: PCA")
for c in np.unique(y_binary): 
    class_mask = (y_binary == c).to_numpy()  
    plt.plot(Z.loc[class_mask, 0], Z.loc[class_mask, 1], "o", alpha=0.5, label=f"Class {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
<<<<<<< Updated upstream
plt.savefig('images/PCA_heart_disease_data.png')

print("PCA done")
=======
plt.savefig('images/PCA_heart_disease_data_binary_wrong.png')
>>>>>>> Stashed changes
