
import pandas as pd
import numpy as np

# =====================================================
# 1. Load noisy dataset
# =====================================================

df = pd.read_csv(
    "dataset/train_noisy.csv",
    dtype={"store": "string", "item": "string"}
)

print("Original shape:", df.shape)


# =====================================================
# 2. Remove exact duplicate rows
# =====================================================

print("\n========== DUPLICATE ROWS ==========")

print("Duplicates before cleaning:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after cleaning:", df.duplicated().sum())


# =====================================================
# 3. Clean date column
# =====================================================

df["date"] = df["date"].str.strip()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# =====================================================
# 4. Convert store, item and sales into numeric
# =====================================================

df["store"] = pd.to_numeric(df["store"], errors="coerce")
df["item"] = pd.to_numeric(df["item"], errors="coerce")
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")


# =====================================================
# 5. Remove invalid store and item values
# =====================================================

df = df[
    df["store"].between(1, 10) &
    df["item"].between(1, 50)
]


# =====================================================
# 6. Handle negative sales
# =====================================================

print("\nNegative sales before cleaning:",
      (df["sales"] < 0).sum())

df.loc[df["sales"] < 0, "sales"] = np.nan


# =====================================================
# 7. Handle missing values
# =====================================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna(subset=["date", "store", "item", "sales"])


# =====================================================
# 8. Inspect extreme sales values
# =====================================================

print("\n========== EXTREME SALES INSPECTION ==========")

print("Maximum sales:", df["sales"].max())

print("Rows with sales equal to 5000:",
      (df["sales"] == 5000).sum())

# 5000 was intentionally introduced as artificial noise
# in our noisy dataset, so remove those records.
df = df[df["sales"] != 5000]


# =====================================================
# 9. Sort the cleaned dataset
# =====================================================

df = df.sort_values(
    by=["date", "store", "item"]
)

df = df.reset_index(drop=True)


# =====================================================
# 10. Save cleaned dataset
# =====================================================

df.to_csv("dataset/train_cleaned.csv", index=False)

print("\n========== CLEANING COMPLETED ==========")

print("Final shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:",
      df.duplicated().sum())

print("\nCleaned dataset saved successfully!")
