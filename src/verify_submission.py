
import pandas as pd
import os

# Load submission file
submission = pd.read_csv("dataset/submission.csv")

print("========== SUBMISSION VERIFICATION ==========")

print("\nFirst 10 rows:")
print(submission.head(10))

print("\nShape:", submission.shape)

print("\nColumns:", submission.columns.tolist())

print("\nMissing values:")
print(submission.isnull().sum())

print("\nDuplicate IDs:", submission["id"].duplicated().sum())

print("\nSales statistics:")
print(submission["sales"].describe())

print("\nFile exists:", os.path.exists("dataset/submission.csv"))
