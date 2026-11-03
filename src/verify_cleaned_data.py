
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/train_cleaned.csv")

print("========== CLEANED DATA VERIFICATION ==========")

# 1. Dataset shape
print("\nDataset shape:", df.shape)

# 2. First five records
print("\nFirst five records:")
print(df.head())

# 3. Data types
print("\nData types:")
print(df.dtypes)

# 4. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# 6. Store validation
print("\nUnique stores:", sorted(df["store"].unique()))

# 7. Item validation
print("\nNumber of unique items:", df["item"].nunique())

# 8. Sales validation
print("\nMinimum sales:", df["sales"].min())
print("Maximum sales:", df["sales"].max())

# 9. Date range
print("\nStart date:", df["date"].min())
print("End date:", df["date"].max())

print("\n========== VERIFICATION COMPLETED ==========")
