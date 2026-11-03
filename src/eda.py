
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("dataset/train_cleaned.csv")

# Convert date into datetime format
df["date"] = pd.to_datetime(df["date"])

# ==================================
# 1. SALES STATISTICS
# ==================================

print("========== SALES STATISTICS ==========")
print(df["sales"].describe())


# ==================================
# 2. SALES DISTRIBUTION
# ==================================

plt.figure(figsize=(8, 5))

sns.histplot(df["sales"], bins=30, kde=True)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()


# ==================================
# 3. DAILY SALES TREND
# ==================================

daily_sales = df.groupby("date")["sales"].sum()

plt.figure(figsize=(14, 5))

plt.plot(daily_sales.index, daily_sales.values)

plt.title("Daily Total Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print("\nDaily sales analysis completed!")

# ==================================
# 4. MONTHLY SALES ANALYSIS
# ==================================

# Extract month and year
df["year_month"] = df["date"].dt.to_period("M")

# Calculate total sales for each month
monthly_sales = df.groupby("year_month")["sales"].sum()

# Plot monthly sales
plt.figure(figsize=(14, 5))

plt.plot(monthly_sales.index.astype(str), monthly_sales.values)

plt.title("Monthly Total Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

# ==================================
# 5. STORE-WISE SALES ANALYSIS
# ==================================

# Calculate total sales for each store
store_sales = df.groupby("store")["sales"].sum()

print("\n========== STORE-WISE SALES ==========")
print(store_sales)

# Plot store-wise sales
plt.figure(figsize=(10, 5))

sns.barplot(
    x=store_sales.index,
    y=store_sales.values
)

plt.title("Total Sales by Store")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")

plt.show()

# ==================================
# 6. ITEM-WISE SALES ANALYSIS
# ==================================

# Calculate total sales for each item
item_sales = df.groupby("item")["sales"].sum()

print("\n========== ITEM-WISE SALES ==========")
print(item_sales)

# Plot item-wise sales
plt.figure(figsize=(14, 5))

sns.barplot(
    x=item_sales.index,
    y=item_sales.values
)

plt.title("Total Sales by Item")
plt.xlabel("Item Number")
plt.ylabel("Total Sales")

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

# ==================================
# 7. WEEKLY SALES ANALYSIS
# ==================================

# Extract day of week
df["day_of_week"] = df["date"].dt.day_name()

# Calculate total sales for each day
weekly_sales = df.groupby("day_of_week")["sales"].sum()

# Arrange days in correct order
day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekly_sales = weekly_sales.reindex(day_order)

print("\n========== WEEKLY SALES ==========")
print(weekly_sales)

# Plot weekly sales
plt.figure(figsize=(10, 5))

sns.barplot(
    x=weekly_sales.index,
    y=weekly_sales.values
)

plt.title("Total Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
# WEEKLY SALES ANALYSIS

df["day_of_week"] = df["date"].dt.day_name()

weekly_sales = df.groupby("day_of_week")["sales"].mean()

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekly_sales = weekly_sales.reindex(day_order)

print("\n========== AVERAGE SALES BY DAY ==========")
print(weekly_sales)

plt.figure(figsize=(10, 5))
sns.barplot(x=weekly_sales.index, y=weekly_sales.values)

plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



