import pandas as pd

# Firstly we read the data and create a dataframe using pandas

data = pd.read_csv('processed_cleveland.csv')

print(data.head())

df = pd.DataFrame(data)
#for checking the amount of ? in the data
print("amount ? in the data =", (df == "?").sum().sum()) 



#print("amount of ? count=", df.count("?"))
## We iterate through the data and change any ineligible datapoints with 0.0, this value might need to be changed later
for row in df.iterrows():
    for val in row:
            df.replace(('?',''), 0.0, inplace=True)

print(df)


# We save the cleaned data to a new csv file

df.to_csv('cleaned_cleveland.csv')

# We do a small safety check to ensure no off values

safetyCheck = open('cleaned_cleveland.csv')

weirdCount = 0

for line in safetyCheck:
    for val in line.split(','):
        if val == '?':
            weirdCount += 1


print("amount of ?=", weirdCount)