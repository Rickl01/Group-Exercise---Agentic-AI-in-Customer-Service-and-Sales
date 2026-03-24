"""
Extra Credit: Housing Demand Forecasting Tool
==============================================
This script predicts housing demand for the next 6 months using
Linear Regression based on historical sales data.

Data Source: Synthetically generated historical housing sales data
             simulating monthly demand patterns with seasonal trends.
             Represents typical real estate market fluctuations.

Author: AI Agentics Assignment

ASSUMPTIONS:
1. Housing demand follows linear trends with seasonal variations
2. Historical patterns will continue into the forecast period
3. No major economic disruptions or policy changes
4. Data represents a single market/region

CHALLENGES:
1. Real estate markets are affected by many external factors
2. Seasonal patterns may shift year-to-year
3. Linear models may not capture complex market dynamics
4. Limited historical data reduces prediction accuracy

POTENTIAL IMPROVEMENTS:
1. Use more sophisticated models (ARIMA, Prophet, LSTM)
2. Include external features (interest rates, employment, GDP)
3. Incorporate seasonal decomposition
4. Use ensemble methods for more robust predictions
5. Add confidence intervals to forecasts
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Generate Historical Housing Demand Data (or load from CSV)
# =============================================================================
# Data Source Comment: Synthetically generated monthly housing sales data
# spanning 36 months with realistic seasonal patterns and growth trends.

def generate_historical_data():
    """Generate 36 months of historical housing demand data."""
    
    # Create date range (3 years of monthly data)
    dates = pd.date_range(start='2022-01-01', periods=36, freq='MS')
    
    # Base demand with upward trend
    base_demand = 100
    trend = np.linspace(0, 30, 36)  # Gradual increase over time
    
    # Seasonal pattern (higher in spring/summer, lower in winter)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(36) / 12 - np.pi/2)
    
    # Random noise
    noise = np.random.normal(0, 8, 36)
    
    # Combine components
    demand = base_demand + trend + seasonal + noise
    demand = np.maximum(demand, 50)  # Minimum demand floor
    
    data = pd.DataFrame({
        'date': dates,
        'month': np.arange(1, 37),  # Month number for regression
        'demand': demand.astype(int)
    })
    
    return data

# Generate data (in practice, you would load from CSV)
print("=" * 70)
print("HOUSING DEMAND FORECASTING TOOL")
print("=" * 70)

# Check if CSV exists, otherwise generate data
csv_filename = 'housing_demand_data.csv'
try:
    df = pd.read_csv(csv_filename)
    print(f"\n✅ Loaded historical data from '{csv_filename}'")
except FileNotFoundError:
    print(f"\n📊 Generating synthetic historical data...")
    df = generate_historical_data()
    df.to_csv(csv_filename, index=False)
    print(f"✅ Data saved to '{csv_filename}'")

# Ensure date column is datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='MS')

# Ensure month column exists
if 'month' not in df.columns:
    df['month'] = np.arange(1, len(df) + 1)

print(f"\nDataset Size: {len(df)} months of historical data")
print(f"\nHistorical Data Preview:")
print(df.head(10).to_string(index=False))

print(f"\nData Statistics:")
print(df['demand'].describe().round(2))

# =============================================================================
# Data Preprocessing
# =============================================================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Feature: Month number (for trend capture)
X = df[['month']].values
y = df['demand'].values

print(f"\nFeature (X): Month number [1, 2, 3, ..., {len(df)}]")
print(f"Target (y): Housing demand (units sold)")

# =============================================================================
# Train Linear Regression Model
# =============================================================================
print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

# Train on all historical data
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"\nLinear Regression Equation:")
print(f"  Demand = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Month")
print(f"\nInterpretation:")
print(f"  - Base demand (month 0): {model.intercept_:.2f} units")
print(f"  - Monthly trend: {model.coef_[0]:+.2f} units/month")
if model.coef_[0] > 0:
    print(f"  - Housing demand is INCREASING over time")
else:
    print(f"  - Housing demand is DECREASING over time")

# Training performance
y_train_pred = model.predict(X)
train_r2 = r2_score(y, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
train_mae = mean_absolute_error(y, y_train_pred)

print(f"\nTraining Performance:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f} units")
print(f"  MAE: {train_mae:.2f} units")

# =============================================================================
# Forecast Next 6 Months
# =============================================================================
print("\n" + "=" * 70)
print("DEMAND FORECAST - NEXT 6 MONTHS")
print("=" * 70)

# Create future months
last_month = df['month'].max()
last_date = df['date'].max()

future_months = np.arange(last_month + 1, last_month + 7).reshape(-1, 1)
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

# Predict
future_demand = model.predict(future_months)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'date': future_dates,
    'month': future_months.flatten(),
    'predicted_demand': future_demand.round(0).astype(int)
})

print("\n6-Month Demand Forecast:")
print("-" * 50)
print(f"{'Date':<15} {'Month #':<10} {'Predicted Demand':<20}")
print("-" * 50)
for _, row in forecast_df.iterrows():
    print(f"{row['date'].strftime('%Y-%m'):<15} {row['month']:<10} {row['predicted_demand']:<20}")

print(f"\nForecast Summary:")
print(f"  Total forecasted demand (6 months): {forecast_df['predicted_demand'].sum():,} units")
print(f"  Average monthly demand: {forecast_df['predicted_demand'].mean():.0f} units")
print(f"  Min forecast: {forecast_df['predicted_demand'].min()} units")
print(f"  Max forecast: {forecast_df['predicted_demand'].max()} units")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Historical Data + Forecast
ax1 = axes[0]
ax1.plot(df['date'], df['demand'], 'b-o', label='Historical Demand', markersize=5)
ax1.plot(df['date'], y_train_pred, 'g--', label='Trend Line (Training)', linewidth=2)
ax1.plot(forecast_df['date'], forecast_df['predicted_demand'], 'r-s', 
         label='Forecasted Demand', markersize=8, linewidth=2)

# Connect historical to forecast
ax1.plot([df['date'].iloc[-1], forecast_df['date'].iloc[0]], 
         [y_train_pred[-1], forecast_df['predicted_demand'].iloc[0]], 
         'g--', linewidth=2)

ax1.axvline(x=df['date'].iloc[-1], color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Housing Demand (Units)', fontsize=12)
ax1.set_title('Housing Demand: Historical Data and 6-Month Forecast', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Forecast Only with Confidence Band
ax2 = axes[1]

# Simple confidence band (±1 RMSE)
lower_bound = forecast_df['predicted_demand'] - train_rmse
upper_bound = forecast_df['predicted_demand'] + train_rmse

ax2.fill_between(forecast_df['date'], lower_bound, upper_bound, 
                  color='red', alpha=0.2, label='Confidence Band (±1 RMSE)')
ax2.plot(forecast_df['date'], forecast_df['predicted_demand'], 'r-s', 
         label='Forecasted Demand', markersize=10, linewidth=2)

# Annotate each forecast point
for _, row in forecast_df.iterrows():
    ax2.annotate(f"{row['predicted_demand']}", 
                 (row['date'], row['predicted_demand']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Predicted Demand (Units)', fontsize=12)
ax2.set_title('6-Month Demand Forecast with Confidence Band', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('demand_forecast_plot.png', dpi=150)
plt.close()

print(f"✅ Forecast visualization saved as 'demand_forecast_plot.png'")

# =============================================================================
# Save Forecast Results
# =============================================================================
forecast_output = 'demand_forecast_results.csv'
forecast_df.to_csv(forecast_output, index=False)
print(f"✅ Forecast results saved to '{forecast_output}'")

# =============================================================================
# Assumptions, Challenges, and Improvements
# =============================================================================
print("\n" + "=" * 70)
print("DOCUMENTATION: ASSUMPTIONS, CHALLENGES, AND IMPROVEMENTS")
print("=" * 70)

print("""
📋 ASSUMPTIONS:

1. LINEAR TREND ASSUMPTION
   - We assume housing demand follows a roughly linear trend over time
   - This simplifies complex market dynamics but may miss non-linear patterns
   
2. HISTORICAL PATTERNS CONTINUE
   - The model assumes past trends will persist into the future
   - No consideration of upcoming market changes or disruptions
   
3. SINGLE MARKET/REGION
   - Data represents one geographic area
   - Results may not generalize to other markets
   
4. NO EXTERNAL FACTORS
   - Model doesn't include interest rates, economic indicators, or policy changes
   - These factors significantly impact real housing markets

⚠️ CHALLENGES:

1. SEASONALITY
   - Real estate has strong seasonal patterns (spring/summer peaks)
   - Linear regression doesn't capture cyclical patterns well
   - Solution: Use seasonal decomposition or time-series models
   
2. EXTERNAL FACTORS
   - Interest rates, employment, GDP affect housing demand
   - These aren't included in our simple model
   - Solution: Add external features to the model
   
3. LIMITED HISTORICAL DATA
   - Only 36 months of data may not capture long-term trends
   - More data would improve model reliability
   
4. MODEL COMPLEXITY
   - Linear regression is too simple for complex market dynamics
   - Non-linear relationships are missed
   
5. UNCERTAINTY QUANTIFICATION
   - Simple confidence bands may underestimate true uncertainty
   - Real forecasts should include proper prediction intervals

🚀 POTENTIAL IMPROVEMENTS:

1. ADVANCED TIME-SERIES MODELS
   - ARIMA/SARIMA: Captures trends and seasonality
   - Facebook Prophet: Handles holidays and trend changes
   - LSTM Neural Networks: Captures complex non-linear patterns
   
2. ADDITIONAL FEATURES
   - Interest rates (mortgage rates)
   - Unemployment rate
   - GDP growth
   - Housing inventory levels
   - Population growth
   - Consumer confidence index
   
3. SEASONAL DECOMPOSITION
   - Separate trend, seasonality, and residual components
   - Forecast each component separately
   
4. ENSEMBLE METHODS
   - Combine multiple models for robust predictions
   - Use weighted averages based on recent performance
   
5. CROSS-VALIDATION
   - Time-series cross-validation for better performance estimates
   - Walk-forward validation
   
6. CONFIDENCE INTERVALS
   - Use proper statistical methods for prediction intervals
   - Bootstrap methods for uncertainty estimation
   
7. REAL-TIME UPDATES
   - Retrain model as new data becomes available
   - Implement online learning for continuous improvement
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
This housing demand forecasting tool:

📊 DATA:
   - Uses {len(df)} months of historical data
   - Data saved to '{csv_filename}'

🤖 MODEL:
   - Linear Regression: Demand = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Month
   - R² Score: {train_r2:.4f}
   - RMSE: {train_rmse:.2f} units

📈 FORECAST (Next 6 Months):
   - Total: {forecast_df['predicted_demand'].sum():,} units
   - Monthly Average: {forecast_df['predicted_demand'].mean():.0f} units
   - Trend: {'Increasing' if model.coef_[0] > 0 else 'Decreasing'}

📁 OUTPUT FILES:
   - Historical data: '{csv_filename}'
   - Forecast results: '{forecast_output}'
   - Visualization: 'demand_forecast_plot.png'

⚡ KEY TAKEAWAY:
   The model predicts {'increasing' if model.coef_[0] > 0 else 'decreasing'} demand
   with approximately {abs(model.coef_[0]):.1f} units change per month.
   For production use, consider more sophisticated models (ARIMA, Prophet, LSTM).
""")
