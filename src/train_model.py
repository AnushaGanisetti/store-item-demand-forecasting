
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load training and validation data
X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    # Save trained model
    joblib.dump(
        model,
        f"models/{name.replace(' ', '_').lower()}.pkl"
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# Display model comparison
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df)

print("\nModel training completed!")
