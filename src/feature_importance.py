
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Feature names
features = [
    "store", "item", "year", "month", "day",
    "day_of_week", "is_weekend",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30"
]

# Extract feature importance
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort in descending order
importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

print("========== FEATURE IMPORTANCE ==========")
print(importance_df)

# Plot graph
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
