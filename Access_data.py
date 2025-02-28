import numpy as np
import pandas as pd

# Firstly we read the data and create a dataframe using pandas

data = pd.read_csv('processed_cleveland.csv')


df = pd.DataFrame(data)

## We iterate through the data and change any ineligible datapoints with 0.0, this value might need to be changed later

for row in df.iterrows():
    for val in row:
        df.replace(('?',''), 0.0, inplace=True)


# We save the cleaned data to a new csv file

df.to_csv('cleaned_cleveland.csv')

# We do a small safety check to ensure no off values

safetyCheck = pd.read_csv('cleaned_cleveland.csv')

