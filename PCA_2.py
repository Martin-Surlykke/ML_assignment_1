import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "cleaned_cleveland.csv"
df = pd.read_csv(file_path)

# Exclude the target variable ("num") for PCA
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
X = df[features]
y = df["num"]  # Target variable

# Handle missing values before transformations
X = X.fillna(X.mean())  # Replace NaN with mean (alternative: median)

# Identify skewed features for log transformation
skewed_features = ["oldpeak", "ca"]

# Ensure there are no negative values before applying log transformation
for feature in skewed_features:
    X[feature] = np.where(X[feature] < 0, 0, X[feature])  # Replace negative values with 0
    X.loc[:, feature] = np.log1p(X[feature])  # Apply log(1+x) transformation

# Standardize the data (mean = 0, variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check for remaining NaN or inf values before PCA
if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
    raise ValueError("X_scaled contains NaN or Inf values after preprocessing!")

# Compute PCA using SVD
U, S, Vh = svd(X_scaled, full_matrices=False)
V = Vh.T  # V from Vh

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# Set threshold for optimal components (90% variance explained)
threshold = 0.9

# Plot variance explained
plt.figure(figsize=(8,5))
plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual Variance")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative Variance")
plt.axhline(y=threshold, color="k", linestyle="--", label="90% Threshold")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.legend()
plt.title("Variance explained by Principal Components")
plt.grid()
plt.savefig('Variance_explained_by_principal_components.png')
plt.show()

# Find the number of components explaining 90% variance
n_components_90 = np.argmax(np.cumsum(rho) >= threshold) + 1
print(f"Number of components explaining 90% variance: {n_components_90}")

# Project data onto principal component space
Z = X_scaled @ V

# Plot PCA (First two principal components)
plt.figure(figsize=(8, 6))
for c in np.unique(y):
    class_mask = (y == c).to_numpy()
    plt.scatter(Z[class_mask, 0], Z[class_mask, 1], alpha=0.5, label=f"Class {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA - Heart Disease Data")
plt.savefig('PCA_heart_disease_data.png')
plt.show()

