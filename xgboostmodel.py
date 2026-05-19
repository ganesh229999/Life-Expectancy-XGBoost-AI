import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # <--- NEW MODEL IMPORT
from sklearn.metrics import r2_score

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
print("🚀 Loading and cleaning data...")
df_raw = pd.read_csv('Life_Expectancy_Data.csv')
df_raw.columns = df_raw.columns.str.strip()

df = df_raw.copy()
df = df.drop(['infant deaths', 'Country', 'Year'], axis=1)
df = df.fillna(df.median(numeric_only=True))
df['Status'] = (df['Status'] == 'Developed').astype(int)

X = df.drop('Life expectancy', axis=1)
y = df['Life expectancy']

# ==========================================
# 2. DATA SPLITTING & SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==========================================
# 3. MODEL COMPARISON (INCLUDING XGBOOST)
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Poisson Regression": PoissonRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost (Extreme)": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

results = {}
model_preds_dict = {}

print("\n--- Model Performance Comparison ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    model_preds_dict[name] = predictions
    
    accuracy = r2_score(y_test, predictions) * 100
    results[name] = accuracy
    print(f"{name}: {accuracy:.2f}%")

# Identify the Best Model automatically
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model Identified: {best_model_name}")

# ==========================================
# 4. VISUALIZATIONS
# ==========================================
print("\n📊 Generating Plots...")

# --- PLOT 1: Scatter Prediction Comparison (2x2 Grid for 4 Models) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Accuracy Comparison: Actual vs Predicted', fontsize=20)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

for i, (name, preds) in enumerate(model_preds_dict.items()):
    sns.scatterplot(x=y_test, y=preds, ax=axes_flat[i], alpha=0.5, color='darkcyan')
    axes_flat[i].plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
    axes_flat[i].set_title(f"{name}\nAccuracy: {results[name]:.2f}%", fontsize=14)
    axes_flat[i].set_xlabel("Actual Life Expectancy")
    axes_flat[i].set_ylabel("Predicted Life Expectancy")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_comparison_scatter.png')


# --- PLOT 2: Country Comparison (Sample of 20) with Country Names ---
sample_indices = y_test.head(20).index
country_names = df_raw.loc[sample_indices, 'Country']

comparison_df = pd.DataFrame({
    'Actual': y_test.head(20).values,
    'Predicted': model_preds_dict[best_model_name][:20]
}, index=country_names)

ax = comparison_df.plot(kind='bar', figsize=(14, 7), color=['#3498db', '#e67e22'], width=0.8)
plt.title(f"Sample Comparison: {best_model_name}", fontsize=16)
plt.ylabel("Years")
plt.xlabel("Country Name")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('country_sample_comparison.png')


# --- PLOT 3: Feature Importance (Best Model) ---
plt.figure(figsize=(10, 8))
if hasattr(best_model, 'feature_importances_'):
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
elif hasattr(best_model, 'coef_'):
    importance = pd.Series(best_model.coef_, index=X.columns).sort_values()

importance.plot(kind='barh', color='teal')
plt.title(f"Key Drivers - {best_model_name}", fontsize=14)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png')

print("✅ Plots Saved: 'model_comparison_scatter.png', 'country_sample_comparison.png', 'feature_importance.png'")

# ==========================================
# 5. SMART PREDICTION FUNCTION
# ==========================================
def smart_predict(schooling, gdp, bmi, alcohol, mortality):
    """Predicts life expectancy based on user input, using the Best Model."""
    input_row = X_train.median().to_dict()
    input_row['Schooling'] = schooling
    input_row['GDP'] = gdp
    input_row['BMI'] = bmi
    input_row['Alcohol'] = alcohol
    input_row['Adult Mortality'] = mortality
    
    input_df = pd.DataFrame([input_row])
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=X.columns)
    pred = best_model.predict(input_scaled)[0]
    return pred

print("\n--- Custom Predictions (Using Best Model) ---")
high_income = smart_predict(schooling=18, gdp=55000, bmi=25, alcohol=2, mortality=50)
low_income = smart_predict(schooling=6, gdp=400, bmi=18, alcohol=0.5, mortality=250)

print(f"🌍 High-Resource Scenario: {high_income:.2f} years")
print(f"🌍 Low-Resource Scenario: {low_income:.2f} years")

# --- SAVE THE ASSETS ---
joblib.dump(best_model, 'life_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# We save the medians to fill in the missing features the user doesn't type in
joblib.dump(X_train.median().to_dict(), 'medians.pkl') 
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ 'life_model.pkl', 'scaler.pkl', 'medians.pkl', and 'feature_names.pkl' created!")