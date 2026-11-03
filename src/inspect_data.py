import pandas as pd

# Load noisy dataset
df = pd.read_csv("dataset/train_noisy.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== BASIC INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\n========== MISSING VALUE PERCENTAGE ==========")

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

print(missing_percentage.round(2))

print("\n========== ROWS WITH MISSING SALES ==========")

missing_sales = df[df["sales"].isnull()]

print(missing_sales.head(20))

print("\nNumber of rows with missing sales:")
print(len(missing_sales))

print("\n========== DUPLICATE ROWS ==========")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

print("\n========== DUPLICATE STORE + ITEM + DATE ==========")

duplicate_keys = df.duplicated(
    subset=["date", "store", "item"]
).sum()

print("Duplicate store-item-date rows:", duplicate_keys)