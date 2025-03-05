import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Firstly we read the data and create a dataframe using pandas

data = pd.read_csv('processed_cleveland.csv')

print(data.head())

print(data.describe())

df = pd.DataFrame(data)



#Checking for missing value
print("Amount of missing values in the different attributes", (df == "?").sum()) 

#Handling missing values by applying mode impuation 
df["ca"] = df["ca"].replace("?", df["ca"].mode()[0])
df["thal"] = df["thal"].replace("?", df["thal"].mode()[0])
#making attribute float64
df["ca"] = pd.to_numeric(df["ca"], errors="coerce")  
df["cp"] = pd.to_numeric(df["cp"], errors="coerce")
df["thal"] = pd.to_numeric(df["thal"], errors="coerce")  

# apply replacements
df["cp"] = df["cp"].replace({4: 0})
df["thal"] = df["thal"].replace({3: 0, 6: 1, 7: 2})

print("ny")
print("most common value in ca = ",df["ca"].mode()[0])
print("most common value in thal =",df["thal"].mode()[0])
#Checking that there is no longer any missing values in the attribues
print("Amount of missing values in the different attributes after imputation", (df == "?").sum())

<<<<<<< Updated upstream
#Checking the data types of the different attributes and insuring they are all float64
print("datatype for collumn that hasent been imputed: oldpeak type:", df["oldpeak"].dtype)
#checing the imputed columns data type
print("ca type:", df["ca"].dtype)
print("thal type:", df["thal"].dtype)

#Changing the imputed attributes types to float64
df["ca"] = pd.to_numeric(df["ca"], errors="coerce")  
df["thal"] = pd.to_numeric(df["thal"], errors="coerce")  
print("ca new type", df["ca"].dtype)  
print("thal new type", df["thal"].dtype)
=======

>>>>>>> Stashed changes



median_ca=df["ca"].median()
median_thal=df["thal"].median()
print("median of ca",median_ca)
print("median of thal",median_thal)
#Histogram of the different attributes
plt.figure(figsize=(20,20), dpi=300)
plt.suptitle('Histogram of the different attributes', fontsize=20)
plt.tight_layout()
df.hist()
plt.savefig('histogram_attributes.png')

# We save the cleaned data to a new csv file
df.to_csv('cleaned_cleveland.csv')



