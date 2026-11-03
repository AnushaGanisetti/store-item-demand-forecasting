
import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("dataset/train_features.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data chronologically
df = df.sort_values("date")

# Select the last 90 days for validation
validation_start = df["date"].max() - pd.Timedelta(days=89)

train_data = df[df["date"] < validation_start]
test_data = df[df["date"] >= validation_start]

# Separate input features and target
features = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

X_train = train_data[features]
y_train = train_data["sales"]

X_test = test_data[features]
y_test = test_data["sales"]

# Display results
print("========== TRAINING DATA ==========")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\n========== VALIDATION DATA ==========")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTraining date range:")
print(train_data["date"].min(), "to", train_data["date"].max())

print("\nValidation date range:")
print(test_data["date"].min(), "to", test_data["date"].max())

# Save split datasets
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)
y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nTrain-test splitting completed!")
