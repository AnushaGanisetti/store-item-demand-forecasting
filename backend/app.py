
from flask import Flask, jsonify, request, render_template
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask application
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# Load test data and predictions
test = pd.read_csv(BASE_DIR / "dataset" / "test.csv")
submission = pd.read_csv(BASE_DIR / "dataset" / "submission.csv")

# Combine test details with predicted sales
forecast_data = test.merge(
    submission,
    on="id",
    how="inner",
    validate="one_to_one"
)

forecast_data["date"] = pd.to_datetime(forecast_data["date"])

# Home route
@app.route("/")
def home():
    return render_template("index.html")
# Health check
@app.route("/api/health")
def health():
    return jsonify({
        "status": "success",
        "forecast_records": len(forecast_data)
    })
# Forecast API
@app.route("/api/forecast")
def forecast():

    date = request.args.get("date")
    store = request.args.get("store")
    item = request.args.get("item")

    if not date or not store or not item:
        return jsonify({
            "error": "Please provide date, store and item"
        }), 400

    try:
        date = pd.to_datetime(date)
        store = int(store)
        item = int(item)
    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid date, store or item"
        }), 400

    result = forecast_data[
        (forecast_data["date"] == date) &
        (forecast_data["store"] == store) &
        (forecast_data["item"] == item)
    ]

    if result.empty:
        return jsonify({
            "error": "No forecast found for the selected details"
        }), 404

    prediction = float(result.iloc[0]["sales"])

    return jsonify({
        "date": date.strftime("%Y-%m-%d"),
        "store": store,
        "item": item,
        "predicted_sales": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
