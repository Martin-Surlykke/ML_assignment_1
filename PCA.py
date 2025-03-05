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
import Correlation_matrix as cm

# read the data and create a dataframe using pandas
data = pd.read_csv('cleaned_cleveland.csv')


#Saving the amount of observations
N = len(data)

#Saving the "num" as the class label
y = data["num"]

#Converting the class label to binary
y_binary = y.apply(lambda x: 1 if x > 0 else 0)

print(y_binary)


#Saving the different categories of the class label
C = len(y_binary.unique())



print("C",C)
print("y",y)
print("y done")
#standardizing the data (or normalizing it?)
# Subtract mean collumn value from each element in each collumn
stand_data = cm.normalize_data(data)
stand_data = cm.extract_relevant_vals(stand_data)

mean_stand_data = stand_data.mean(axis=0)
print("mean_stand_data")
print(mean_stand_data)

#PCA by computing SVD of stand_data

U, S, Vh = svd(stand_data, full_matrices=False)

#Get V from Vh = V.hermitian
V = Vh.T

dot_product = np.dot(V[:, 0], V[:, 1])
print(f"Dot product of PC1 and PC2: {dot_product:.5f}")  # Should be close to 0

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



colorList = ["green", "purple"]

for i in range(2):
    plt.arrow(0, 0, V[i, 0], V[i, 1], color=colorList[i], alpha=0.7, width=0.1)


plt.savefig('images/PCA_heart_disease_data_binary_directions.png')

# Plot PCA of the data
f = plt.figure()
plt.title("Heart Disease data: PCA")
for c in np.unique(y):
    class_mask = (y == c).to_numpy()
    plt.plot(Z.loc[class_mask, 0], Z.loc[class_mask, 1], "o", alpha=0.5, label=f"Class {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig('images/PCA_heart_disease_data.png')

# plot 3D PCA of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("Heart Disease data: PCA")
for c in np.unique(y_binary):
    class_mask = (y_binary == c).to_numpy()
    ax.scatter(Z.loc[class_mask, 0], Z.loc[class_mask, 1], Z.loc[class_mask, 2], alpha=0.4, label=f"Class {c}")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
# change the viewing angle
ax.view_init(elev=0, azim=270)

plt.legend()
plt.savefig('images/PCA_heart_disease_data_3D_directions.png')


