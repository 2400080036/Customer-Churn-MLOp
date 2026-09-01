import pandas as pd

data = pd.read_csv("data/customer_churn.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nChurn count:")
print(data["churn"].value_counts())