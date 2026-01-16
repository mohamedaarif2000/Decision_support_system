import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_excel(r"C:\Users\Mohamedaarif\PycharmProjects\PythonProject\Dataset\Renewable_DSS_All_Data.xlsx")

# Correct target
target = "Demand"  # Assuming you want to predict energy demand/output

# Features and target
X = df.drop(columns=[target])
y = df[target]

# ================================
# Statistical Model (OLS)
# ================================
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()
print("=== OLS Summary ===")
print(ols_model.summary())

# Save OLS summary to file
with open("output/OLS_summary.txt", "w") as f:
    f.write(str(ols_model.summary()))

# ================================
# Machine Learning Models
# ================================

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Linear Regression ----
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ---- Random Forest ----
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ---- Support Vector Regressor ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svr = SVR(kernel="rbf")
svr.fit(X_train_s, y_train_s)
svr_pred = svr.predict(X_test_s)

# R² Scores
print("\n=== Model Performance ===")
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("SVR R2:", r2_score(y_test_s, svr_pred))

# Save results
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "SVR"],
    "R2_Score": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test_s, svr_pred)
    ]
})
results.to_csv("output/model_performance.csv", index=False)

# Feature Importance from Random Forest
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\n=== Feature Importance ===")
print(importance)
importance.to_csv("output/feature_importance.csv", index=False)
