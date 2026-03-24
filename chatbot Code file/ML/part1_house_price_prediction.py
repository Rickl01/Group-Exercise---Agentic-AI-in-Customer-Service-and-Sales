"""
Part 1: House Price Prediction using Linear Regression
=======================================================
This script predicts house prices based on square footage and location
using a Linear Regression model with scikit-learn.

Data Source: Synthetically generated dataset based on realistic US housing market trends.
             Prices reflect approximate 2024 market values with location-based variations.
             Generated programmatically to simulate real-world housing data patterns.

Author: AI Agentics Assignment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Generate Realistic Dataset (100+ records)
# =============================================================================
# Data Source Comment: Synthetically generated based on US housing market trends.
# Downtown properties have higher base prices, Rural properties are more affordable,
# and Suburb properties fall in between. Price per square foot varies by location.

n_samples = 150  # 150 records for robust training

# Generate locations with realistic distribution
locations = np.random.choice(
    ['Downtown', 'Suburb', 'Rural'], 
    size=n_samples, 
    p=[0.3, 0.5, 0.2]  # 30% Downtown, 50% Suburb, 20% Rural
)

# Generate square footage (realistic range: 800 - 4000 sq ft)
square_footage = np.random.randint(800, 4001, size=n_samples)

# Generate prices based on location and square footage with realistic coefficients
# Base price + (price per sq ft * sq ft) + noise
def calculate_price(sqft, location):
    """Calculate house price based on location and square footage."""
    # Base prices by location (reflects land value and demand)
    base_prices = {
        'Downtown': 150000,  # Higher base due to location premium
        'Suburb': 100000,    # Moderate base price
        'Rural': 50000       # Lower base due to less demand
    }
    
    # Price per square foot by location
    price_per_sqft = {
        'Downtown': 250,  # $250/sqft - premium urban pricing
        'Suburb': 180,    # $180/sqft - moderate pricing
        'Rural': 120      # $120/sqft - affordable pricing
    }
    
    base = base_prices[location]
    rate = price_per_sqft[location]
    
    # Add realistic noise (±10% variance)
    noise = np.random.normal(0, 0.10 * (base + rate * sqft))
    
    price = base + (rate * sqft) + noise
    return max(price, 50000)  # Minimum price floor

# Generate prices
prices = [calculate_price(sqft, loc) for sqft, loc in zip(square_footage, locations)]

# Create DataFrame
data = {
    'square_footage': square_footage,
    'location': locations,
    'price': np.round(prices, -2)  # Round to nearest $100
}

df = pd.DataFrame(data)

# =============================================================================
# Display Dataset Info
# =============================================================================
print("=" * 60)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("=" * 60)
print(f"\nDataset Size: {len(df)} records")
print(f"\nDataset Preview (first 10 rows):")
print(df.head(10).to_string(index=False))

print(f"\nDataset Statistics:")
print(df.describe().round(2))

print(f"\nLocation Distribution:")
print(df['location'].value_counts())

# =============================================================================
# Prepare Features and Target
# =============================================================================
# Features: square_footage (numeric) and location (categorical)
X = df[['square_footage', 'location']]
# Target: price
y = df['price']

# =============================================================================
# Preprocessing Pipeline
# =============================================================================
# Use OneHotEncoder to convert categorical 'location' into numeric features
# This creates binary columns: location_Downtown, location_Rural, location_Suburb

preprocessor = ColumnTransformer(
    transformers=[
        ('location', OneHotEncoder(sparse_output=False, drop='first'), ['location'])
        # drop='first' avoids multicollinearity (dummy variable trap)
    ],
    remainder='passthrough'  # Keep square_footage as-is
)

# Create pipeline: Preprocessing -> Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# =============================================================================
# Train-Test Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 80% train, 20% test
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# =============================================================================
# Train the Model
# =============================================================================
model.fit(X_train, y_train)

# =============================================================================
# Model Evaluation
# =============================================================================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"R² Score: {r2:.4f} (explains {r2*100:.1f}% of variance)")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")

# =============================================================================
# Model Coefficients - Understanding the Model
# =============================================================================
print("\n" + "=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

# Get feature names after preprocessing
# With drop='first', we have: location_Rural, location_Suburb, square_footage
# (Downtown is the reference/baseline category)
ohe_features = model.named_steps['preprocessor'].named_transformers_['location'].get_feature_names_out(['location']).tolist()
feature_names = ohe_features + ['square_footage']

coefficients = model.named_steps['regressor'].coef_
intercept = model.named_steps['regressor'].intercept_

print(f"\nIntercept (Base price for Downtown): ${intercept:,.2f}")
print("\nCoefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"  {feature}: ${coef:,.2f}")

# =============================================================================
# Coefficient Interpretation
# =============================================================================
print("\n" + "=" * 60)
print("COEFFICIENT INTERPRETATION")
print("=" * 60)

# Find square_footage coefficient
sqft_coef = coefficients[feature_names.index('square_footage')]
print(f"""
1. SQUARE FOOTAGE COEFFICIENT: ${sqft_coef:,.2f}
   
   This means that for every additional square foot of living space,
   the predicted house price increases by approximately ${sqft_coef:,.2f}.
   
   For example:
   - A house with 1000 sq ft vs 1500 sq ft (500 sq ft difference)
   - Price difference ≈ 500 × ${sqft_coef:,.2f} = ${500 * sqft_coef:,.2f}

2. LOCATION EFFECT ON PRICE:
   
   The location coefficients show the price difference relative to Downtown
   (Downtown is the baseline/reference category with coefficient = 0):
""")

for feature, coef in zip(feature_names, coefficients):
    if 'location' in feature:
        location_name = feature.replace('location_', '')
        if coef < 0:
            print(f"   - {location_name}: ${coef:,.2f} (${abs(coef):,.2f} LESS than Downtown)")
        else:
            print(f"   - {location_name}: ${coef:,.2f} (${coef:,.2f} MORE than Downtown)")

print("""
   Downtown typically commands higher prices due to:
   - Proximity to jobs, entertainment, and amenities
   - Higher demand and limited supply
   - Better access to public transportation
   
   Rural areas are typically more affordable due to:
   - Distance from urban centers
   - Lower demand
   - More available land
""")

# =============================================================================
# Prediction for New House
# =============================================================================
print("=" * 60)
print("PREDICTION: 2000 sq ft house in Downtown")
print("=" * 60)

new_house = pd.DataFrame({
    'square_footage': [2000], 
    'location': ['Downtown']
})

predicted_price = model.predict(new_house)
print(f"\nInput: 2000 sq ft house in Downtown")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")

# Additional predictions for comparison
print("\n" + "-" * 40)
print("COMPARISON: Same size house in different locations")
print("-" * 40)

for loc in ['Downtown', 'Suburb', 'Rural']:
    house = pd.DataFrame({'square_footage': [2000], 'location': [loc]})
    pred = model.predict(house)[0]
    print(f"  2000 sq ft in {loc:10}: ${pred:,.2f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
This linear regression model predicts house prices using:
- Square footage (continuous variable)
- Location (categorical variable: Downtown, Suburb, Rural)

Key Findings:
- Each square foot adds approximately ${sqft_coef:,.2f} to the price
- Downtown properties have the highest baseline price
- Rural properties are the most affordable
- The model explains {r2*100:.1f}% of the price variance (R² = {r2:.4f})

Model can be improved by adding features like:
- Number of bedrooms/bathrooms
- Year built
- Lot size
- Condition/renovations
""")
