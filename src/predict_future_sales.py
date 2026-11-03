
import pandas as pd
import numpy as np
import joblib

# Load cleaned training data and test data
train = pd.read_csv("dataset/train_cleaned.csv")
test = pd.read_csv("dataset/test.csv")

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Store historical sales for every store-item combination
sales_history = {}

for (store, item), group in train.groupby(["store", "item"]):
    sales_history[(store, item)] = (
        group.sort_values("date")["sales"].tolist()
    )

# Feature order must match model training
features = [
    "store", "item", "year", "month", "day",
    "day_of_week", "is_weekend",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30"
]

# Store predictions using original test row indices
predictions = {}

# Predict one date at a time
for current_date in sorted(test["date"].unique()):

    daily_data = test[test["date"] == current_date]

    rows = []
    row_indices = []
    keys = []

    for index, row in daily_data.iterrows():

        store = int(row["store"])
        item = int(row["item"])

        past_sales = sales_history[(store, item)]

        features_row = {
            "store": store,
            "item": item,
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "day_of_week": current_date.dayofweek,
            "is_weekend": int(current_date.dayofweek >= 5),
            "lag_1": past_sales[-1],
            "lag_7": past_sales[-7],
            "lag_30": past_sales[-30],
            "rolling_mean_7": np.mean(past_sales[-7:]),
            "rolling_mean_30": np.mean(past_sales[-30:])
        }

        rows.append(features_row)
        row_indices.append(index)
        keys.append((store, item))

    # Prepare input for model
    X_day = pd.DataFrame(rows)[features]

    # Predict sales for the current date
    daily_predictions = np.maximum(model.predict(X_day), 0)

    # Save predictions and update history
    for i, prediction in enumerate(daily_predictions):

        predictions[row_indices[i]] = prediction

        store, item = keys[i]

        # Use predicted sales for future dates
        sales_history[(store, item)].append(prediction)

    print("Predictions completed for:", current_date.date())

# Prepare final submission in original test order
submission = pd.DataFrame({
    "id": test["id"],
    "sales": [predictions[i] for i in test.index]
})

# Save final predictions
submission.to_csv("dataset/submission.csv", index=False)

print("\n========== FINAL PREDICTIONS ==========")
print(submission.head(10))

print("\nTotal predictions:", len(submission))
print("Missing predictions:", submission["sales"].isna().sum())

print("\nSubmission file saved successfully!")
