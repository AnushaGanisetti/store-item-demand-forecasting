
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned historical data
df = pd.read_csv("dataset/train_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])

# Define validation period
validation_start = pd.Timestamp("2017-10-03")

# Separate training history and validation data
history = df[df["date"] < validation_start].copy()
validation = df[df["date"] >= validation_start].copy()

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Store previous sales for every store-item combination
sales_history = {}

for (store, item), group in history.groupby(["store", "item"]):
    sales_history[(store, item)] = list(group.sort_values("date")["sales"])

predictions = []

# Predict one day at a time
for current_date in sorted(validation["date"].unique()):

    daily_data = validation[validation["date"] == current_date]
    rows = []

    for _, row in daily_data.iterrows():

        store = int(row["store"])
        item = int(row["item"])

        past_sales = sales_history[(store, item)]

        features = {
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

        rows.append(features)

    X_day = pd.DataFrame(rows)

    predicted_sales = np.maximum(model.predict(X_day), 0)

    for i, (_, row) in enumerate(daily_data.iterrows()):

        store = int(row["store"])
        item = int(row["item"])

        prediction = predicted_sales[i]

        predictions.append(prediction)

        # Add prediction to history for future dates
        sales_history[(store, item)].append(prediction)

# Evaluate recursive predictions
actual = validation["sales"].values

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("========== RECURSIVE FORECAST RESULTS ==========")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

comparison = pd.DataFrame({
    "Actual Sales": actual[:20],
    "Predicted Sales": predictions[:20]
})

print("\n========== FIRST 20 PREDICTIONS ==========")
print(comparison)
