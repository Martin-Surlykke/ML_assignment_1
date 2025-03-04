import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "cleaned_cleveland.csv"
df = pd.read_csv(file_path)

# Select the most relevant attributes for histogram visualization
hist_columns = ["ca", "oldpeak", "thal", "slope", "thalach"]

# Create histograms for continuous variables
plt.figure(figsize=(12, 8))
df[hist_columns].hist(bins=20, figsize=(12, 8), grid=False, edgecolor='black')
plt.tight_layout()
plt.show()

# Save the histogram figure
histogram_path = "histograms.png"
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Adjust layout for 5 histograms
df[hist_columns].hist(bins=20, grid=False, edgecolor='black', ax=axes.flatten()[:5])
plt.tight_layout()
plt.savefig(histogram_path)
plt.show()

# Bar chart for categorical feature `cp`
cp_fig_path = "cp_bar.png"  # Define the file name

plt.figure(figsize=(6, 4))
df["cp"].value_counts().sort_index().plot(kind="bar", edgecolor='black')
plt.xlabel("Chest Pain Type (cp)")
plt.ylabel("Frequency")
plt.title("Distribution of Chest Pain Types")
plt.xticks([0, 1, 2, 3], ["Type 1", "Type 2", "Type 3", "Type 4"], rotation=0)

# Save the bar chart
plt.tight_layout()
plt.savefig(cp_fig_path)  # This ensures the figure is saved
plt.show()
