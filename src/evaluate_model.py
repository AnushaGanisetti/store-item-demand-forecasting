
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

# Load validation data
X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Predict sales
predictions = model.predict(X_test)

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

print("========== ACTUAL VS PREDICTED ==========")
print(comparison.head(20))

# Calculate MAE
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", mae)

# Plot comparison for first 100 records
plt.figure(figsize=(14, 5))

plt.plot(
    y_test.iloc[:100].values,
    label="Actual Sales"
)

plt.plot(
    predictions[:100],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
