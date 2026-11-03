import pandas as pd
import numpy as np

# Make the random changes reproducible
np.random.seed(42)

# Load the original dataset
df = pd.read_csv("dataset/train.csv")

print("Original dataset:")
print(df.head())
print("\nOriginal shape:", df.shape)


# =====================================================
# 1. Add missing sales values
# =====================================================

missing_indices = np.random.choice(
    df.index,
    size=int(len(df) * 0.01),
    replace=False
)

df.loc[missing_indices, "sales"] = np.nan


# =====================================================
# 2. Add duplicate rows
# =====================================================

duplicate_rows = df.sample(
    n=int(len(df) * 0.005),
    random_state=42
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)


# =====================================================
# 3. Add negative sales values
# =====================================================

negative_indices = np.random.choice(
    df.index,
    size=30,
    replace=False
)

df.loc[negative_indices, "sales"] = -10


# =====================================================
# 4. Add extreme sales outliers
# =====================================================

outlier_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[outlier_indices, "sales"] = 5000
# =====================================================
# Convert store and item to text
# =====================================================

df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)



# =====================================================
# 5. Add invalid store IDs
# =====================================================

invalid_store_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[invalid_store_indices, "store"] = "Store_99"


# =====================================================
# 6. Add invalid item IDs
# =====================================================

invalid_item_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[invalid_item_indices, "item"] = "Item_-5"


# =====================================================
# 7. Add whitespace to some dates
# =====================================================

date_indices = np.random.choice(
    df.index,
    size=30,
    replace=False
)

df.loc[date_indices, "date"] = (
    " " + df.loc[date_indices, "date"].astype(str) + " "
)
# Force store and item to text
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)

# =====================================================
# 8. Save noisy dataset
# =====================================================

df.to_csv(
    "dataset/train_noisy.csv",
    index=False
)

print("\nNoisy dataset created successfully!")
print("Noisy shape:", df.shape)