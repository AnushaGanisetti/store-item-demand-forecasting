\# Store Item Demand Forecasting System



\## Project Overview

The Store Item Demand Forecasting System is a Machine Learning

project that predicts the expected daily sales quantity of

different items across multiple stores.



It uses historical sales data to identify patterns and forecast

future product demand.



\## Objectives

\- Predict daily sales for individual store-item combinations.

\- Analyze historical sales trends.

\- Perform data cleaning and feature engineering.

\- Compare multiple Machine Learning algorithms.

\- Display sales forecasts through a web dashboard.



\## Technologies Used

\- Python

\- Pandas, NumPy

\- Scikit-learn

\- Matplotlib, Seaborn

\- Flask

\- HTML, CSS, JavaScript

\- Git and GitHub



\## Machine Learning Workflow

1\. Data collection

2\. Data inspection and cleaning

3\. Exploratory Data Analysis

4\. Feature engineering

5\. Training and validation split

6\. Model training

7\. Model evaluation

8\. Future sales prediction



\## Algorithms

\- Linear Regression

\- Decision Tree Regressor

\- Random Forest Regressor



\## Model Performance



Random Forest achieved the best one-step validation results:



| Metric | Score |

|---|---:|

| MAE | 6.1459 |

| RMSE | 8.0422 |

| R² Score | 0.9201 |



Recursive 90-day evaluation:



| Metric | Score |

|---|---:|

| MAE | 6.3329 |

| RMSE | 8.3548 |

| R² Score | 0.9138 |



\## Web Application

The Flask application provides a dashboard where users can select:

\- Date

\- Store

\- Item



The application displays the predicted sales quantity.



The dashboard currently retrieves precomputed forecasts

from the submission dataset.



\## Project Structure



```text

Demand Forecasting System/

├── backend/

├── dataset/

├── models/

├── notebooks/

├── src/

├── static/

├── templates/

├── requirements.txt

├── README.md

└── .gitignore

