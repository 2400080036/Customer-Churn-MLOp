"""Quick look at the dataset (run this first to understand the data)."""
from src.data import load_data
from src.config import TARGET

data = load_data()
print("Rows, columns:", data.shape)
print("\nChurn counts:\n", data[TARGET].value_counts())
print(f"\nChurn rate: {(data[TARGET] == 'Yes').mean():.1%}")
print("\nMissing values per column:\n", data.isna().sum()[data.isna().sum() > 0])
print("\nFirst 5 rows:\n", data.head())
