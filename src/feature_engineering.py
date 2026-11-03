
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/train_cleaned.csv")

# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract date-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Display the new columns
print("========== FEATURE ENGINEERING ==========")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns)

# Save the dataset
df.to_csv("dataset/train_features.csv", index=False)

print("\nFeature engineering completed!")

# Sort data by store, item and date
df = df.sort_values(
    by=["store", "item", "date"]
)

# Group sales by each store and item combination
grouped = df.groupby(["store", "item"])["sales"]

# Create lag features
df["lag_1"] = grouped.shift(1)
df["lag_7"] = grouped.shift(7)
df["lag_30"] = grouped.shift(30)

print("\n========== LAG FEATURES ==========")
print(df[[
    "date", "store", "item", "sales",
    "lag_1", "lag_7", "lag_30"
]].head(35))
df.to_csv("dataset/train_features.csv", index=False)

# Create rolling average features
df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

df["rolling_mean_30"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=30).mean())
)

print("\n========== ROLLING AVERAGE FEATURES ==========")

print(df[[
    "date",
    "store",
    "item",
    "sales",
    "rolling_mean_7",
    "rolling_mean_30"
]].head(40))

# Check missing values before handling them
print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())

# Remove rows with missing lag and rolling features
df = df.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30"
    ]
)

# Reset index
df = df.reset_index(drop=True)

# Check missing values after cleaning
print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)

# Save final feature dataset
df.to_csv("dataset/train_features.csv", index=False)

print("\nFinal feature dataset saved successfully!")


