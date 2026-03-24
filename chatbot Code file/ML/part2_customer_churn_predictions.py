"""
Part 2: Customer Churn Prediction using Logistic Regression
============================================================
This script predicts customer churn probability using a Logistic Regression model.
Churn = customers who stop doing business with a company.

Data Source: Synthetically generated dataset based on typical telecom/subscription
             service customer behavior patterns. Features reflect real-world
             indicators of customer churn risk.

Author: AI Agentics Assignment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Generate Realistic Dataset (150+ records)
# =============================================================================
# Data Source Comment: Synthetically generated based on telecom/SaaS industry
# customer behavior patterns. Churn indicators include low usage, high service
# calls, low purchase amounts, and regional factors.

n_samples = 200  # 200 records for robust training

# Generate customer features
ages = np.random.randint(18, 70, size=n_samples)
monthly_usage_hours = np.random.exponential(scale=30, size=n_samples).clip(1, 100).astype(int)
purchase_amounts = np.random.exponential(scale=150, size=n_samples).clip(20, 500).astype(int)
customer_service_calls = np.random.poisson(lam=3, size=n_samples).clip(0, 15)
regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples, p=[0.25, 0.30, 0.20, 0.25])

# Generate churn based on realistic factors
def calculate_churn_probability(age, usage, purchase, calls, region):
    """
    Calculate churn probability based on customer features.
    Higher churn risk for: low usage, low purchase, high service calls
    """
    # Base churn probability
    prob = 0.3
    
    # Usage effect: Low usage increases churn risk
    if usage < 15:
        prob += 0.25
    elif usage < 30:
        prob += 0.10
    elif usage > 50:
        prob -= 0.15
    
    # Purchase amount effect: Low spending increases churn
    if purchase < 80:
        prob += 0.20
    elif purchase < 150:
        prob += 0.05
    elif purchase > 250:
        prob -= 0.15
    
    # Customer service calls: High calls = frustration = churn
    if calls >= 6:
        prob += 0.30
    elif calls >= 4:
        prob += 0.15
    elif calls <= 1:
        prob -= 0.10
    
    # Age effect: Younger customers slightly more likely to switch
    if age < 25:
        prob += 0.08
    elif age > 55:
        prob -= 0.05
    
    # Regional effect (simulating regional service quality differences)
    region_effect = {'North': 0.0, 'South': -0.05, 'East': 0.05, 'West': 0.02}
    prob += region_effect[region]
    
    # Clamp probability
    return max(0.05, min(0.95, prob))

# Generate churn labels based on probability
churn_probs = [calculate_churn_probability(a, u, p, c, r) 
               for a, u, p, c, r in zip(ages, monthly_usage_hours, purchase_amounts, 
                                        customer_service_calls, regions)]
churn = [1 if np.random.random() < prob else 0 for prob in churn_probs]

# Create DataFrame
data = {
    'age': ages,
    'monthly_usage_hours': monthly_usage_hours,
    'purchase_amount': purchase_amounts,
    'customer_service_calls': customer_service_calls,
    'region': regions,
    'churn': churn  # 1 = churned, 0 = retained
}

df = pd.DataFrame(data)

# =============================================================================
# Display Dataset Info
# =============================================================================
print("=" * 65)
print("CUSTOMER CHURN PREDICTION - LOGISTIC REGRESSION")
print("=" * 65)
print(f"\nDataset Size: {len(df)} records")
print(f"\nDataset Preview (first 10 rows):")
print(df.head(10).to_string(index=False))

print(f"\nDataset Statistics:")
print(df.describe().round(2))

print(f"\nChurn Distribution:")
churn_counts = df['churn'].value_counts()
print(f"  Not Churned (0): {churn_counts.get(0, 0)} ({churn_counts.get(0, 0)/len(df)*100:.1f}%)")
print(f"  Churned (1):     {churn_counts.get(1, 0)} ({churn_counts.get(1, 0)/len(df)*100:.1f}%)")

print(f"\nRegion Distribution:")
print(df['region'].value_counts())

# =============================================================================
# Prepare Features and Target
# =============================================================================
# Features: age, monthly_usage_hours, purchase_amount, customer_service_calls, region
X = df[['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls', 'region']]
# Target: churn (1 = churned, 0 = not churned)
y = df['churn']

# =============================================================================
# Preprocessing Pipeline
# =============================================================================
# StandardScaler: Normalizes numerical features (mean=0, std=1)
# OneHotEncoder: Converts categorical 'region' into binary columns

numerical_features = ['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']
categorical_features = ['region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
    ]
)

# Create pipeline: Preprocessing -> Logistic Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# =============================================================================
# Train-Test Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain churn ratio in both sets
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
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}% correct predictions)")
print(f"Precision: {precision:.4f} (of predicted churners, {precision*100:.1f}% actually churned)")
print(f"Recall:    {recall:.4f} (caught {recall*100:.1f}% of actual churners)")
print(f"F1 Score:  {f1:.4f} (harmonic mean of precision & recall)")
print(f"ROC-AUC:   {roc_auc:.4f} (model's ability to distinguish classes)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                  Predicted")
print(f"                  No Churn  Churn")
print(f"  Actual No Churn    {cm[0][0]:3d}      {cm[0][1]:3d}")
print(f"  Actual Churn       {cm[1][0]:3d}      {cm[1][1]:3d}")

# =============================================================================
# Model Coefficients
# =============================================================================
print("\n" + "=" * 65)
print("MODEL COEFFICIENTS")
print("=" * 65)

# Get feature names after preprocessing
# Numerical features keep their names
# Categorical features are one-hot encoded (drop='first' means North is baseline)
ohe_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['region']).tolist()
feature_names = numerical_features + ohe_features

coefficients = model.named_steps['classifier'].coef_[0]
intercept = model.named_steps['classifier'].intercept_[0]

print(f"\nIntercept: {intercept:.4f}")
print("\nFeature Coefficients (impact on churn log-odds):")
for feature, coef in sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑ increases churn" if coef > 0 else "↓ decreases churn"
    print(f"  {feature:25}: {coef:+.4f} ({direction})")

# =============================================================================
# Coefficient Interpretation
# =============================================================================
print("\n" + "=" * 65)
print("COEFFICIENT INTERPRETATION")
print("=" * 65)

# Find key coefficients
service_calls_coef = coefficients[feature_names.index('customer_service_calls')]
usage_coef = coefficients[feature_names.index('monthly_usage_hours')]
purchase_coef = coefficients[feature_names.index('purchase_amount')]

print(f"""
UNDERSTANDING LOGISTIC REGRESSION COEFFICIENTS:

Coefficients represent the change in LOG-ODDS of churn for a 1-unit
increase in the feature (after scaling). Positive = increases churn risk.

KEY INSIGHTS:

1. CUSTOMER SERVICE CALLS: {service_calls_coef:+.4f}
   {"Higher service calls INCREASE churn risk." if service_calls_coef > 0 else "Higher service calls DECREASE churn risk."}
   Customers who contact support frequently are often frustrated
   with the product/service and are more likely to leave.

2. MONTHLY USAGE HOURS: {usage_coef:+.4f}
   {"Higher usage INCREASES churn risk." if usage_coef > 0 else "Higher usage DECREASES churn risk."}
   {"This is unexpected - may indicate heavy users hitting limitations." if usage_coef > 0 else "Engaged customers who use the product regularly are less likely to churn."}

3. PURCHASE AMOUNT: {purchase_coef:+.4f}
   {"Higher spending INCREASES churn risk." if purchase_coef > 0 else "Higher spending DECREASES churn risk."}
   {"This is unexpected - may warrant further investigation." if purchase_coef > 0 else "Customers who spend more have higher investment and are less likely to leave."}

REGIONAL EFFECTS (relative to North baseline):
""")

for feature, coef in zip(feature_names, coefficients):
    if 'region' in feature:
        region_name = feature.replace('region_', '')
        effect = "higher" if coef > 0 else "lower"
        print(f"   - {region_name}: {coef:+.4f} ({effect} churn risk than North)")

# =============================================================================
# Prediction for New Customer
# =============================================================================
print("\n" + "=" * 65)
print("PREDICTION: New Customer Churn Risk Assessment")
print("=" * 65)

new_customer = pd.DataFrame({
    'age': [35],
    'monthly_usage_hours': [20],
    'purchase_amount': [150],
    'customer_service_calls': [5],
    'region': ['West']
})

churn_probability = model.predict_proba(new_customer)[0][1]
threshold = 0.5
churn_prediction = 1 if churn_probability > threshold else 0

print(f"\nNew Customer Profile:")
print(f"  Age: 35 years")
print(f"  Monthly Usage: 20 hours")
print(f"  Purchase Amount: $150")
print(f"  Customer Service Calls: 5")
print(f"  Region: West")

print(f"\n  CHURN PROBABILITY: {churn_probability:.2%}")
print(f"  THRESHOLD: {threshold:.0%}")
print(f"  PREDICTION: {'⚠️  AT RISK OF CHURNING' if churn_prediction == 1 else '✅ LIKELY TO STAY'}")

# =============================================================================
# Business Application
# =============================================================================
print("\n" + "=" * 65)
print("BUSINESS APPLICATION: Using Churn Predictions")
print("=" * 65)

print(f"""
WHAT DOES CHURN PROBABILITY MEAN?

A churn probability of {churn_probability:.0%} means there is a {churn_probability:.0%} chance
this customer will stop doing business with us in the near future.

HOW BUSINESSES CAN USE THIS TO REDUCE CHURN:

1. IDENTIFY AT-RISK CUSTOMERS
   - Run predictions on entire customer base
   - Flag customers with probability > 50% (or lower for proactive approach)
   - Prioritize intervention by probability score

2. TARGETED RETENTION STRATEGIES
   - High Service Calls → Proactive support outreach, issue resolution
   - Low Usage → Re-engagement campaigns, feature education
   - Low Purchase Amount → Special discounts, loyalty rewards

3. INTERVENTION EXAMPLES FOR THIS CUSTOMER (prob={churn_probability:.0%}):
""")

if churn_probability > 0.5:
    print("""   ⚠️  HIGH RISK - Recommended Actions:
   - Assign dedicated account manager
   - Offer retention discount (10-20% off)
   - Schedule check-in call to address concerns
   - Review support ticket history for unresolved issues
   - Consider service upgrade or free trial of premium features""")
else:
    print("""   ✅ MODERATE/LOW RISK - Recommended Actions:
   - Standard engagement campaigns
   - Periodic satisfaction surveys
   - Loyalty rewards for continued business
   - Cross-sell complementary products""")

print(f"""
4. MEASURE IMPACT
   - Track churn rate before/after interventions
   - Calculate Customer Lifetime Value (CLV) saved
   - A/B test different retention strategies

5. CONTINUOUS IMPROVEMENT
   - Retrain model with new data quarterly
   - Add new features (e.g., NPS scores, payment history)
   - Adjust threshold based on business capacity
""")

# =============================================================================
# Batch Prediction Example
# =============================================================================
print("=" * 65)
print("BATCH PREDICTION: Multiple Customer Risk Assessment")
print("=" * 65)

# Create sample batch of customers
batch_customers = pd.DataFrame({
    'age': [22, 45, 38, 55, 29],
    'monthly_usage_hours': [8, 60, 25, 45, 12],
    'purchase_amount': [50, 300, 180, 250, 75],
    'customer_service_calls': [7, 1, 3, 2, 8],
    'region': ['East', 'South', 'North', 'West', 'East']
})

batch_proba = model.predict_proba(batch_customers)[:, 1]
batch_pred = model.predict(batch_customers)

print("\nCustomer Risk Assessment:")
print("-" * 65)
for i, (_, row) in enumerate(batch_customers.iterrows()):
    risk_level = "HIGH" if batch_proba[i] > 0.6 else "MEDIUM" if batch_proba[i] > 0.4 else "LOW"
    status = "⚠️" if batch_pred[i] == 1 else "✅"
    print(f"Customer {i+1}: Age={row['age']:2d}, Usage={row['monthly_usage_hours']:2d}h, "
          f"Calls={row['customer_service_calls']} → Prob={batch_proba[i]:.0%} [{risk_level}] {status}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"""
This logistic regression model predicts customer churn using:
- Age (demographic)
- Monthly Usage Hours (engagement)
- Purchase Amount (spending behavior)
- Customer Service Calls (satisfaction indicator)
- Region (categorical)

Key Findings:
- Model Accuracy: {accuracy*100:.1f}%
- ROC-AUC Score: {roc_auc:.4f}
- Primary churn indicators: service calls, low usage, low spending

Business Value:
- Identify at-risk customers before they leave
- Target retention efforts efficiently
- Reduce customer acquisition costs by retaining existing customers
- Improve customer lifetime value (CLV)

Model can be improved by adding:
- Contract length / tenure
- Payment method
- Complaint history / NPS scores
- Product usage patterns
- Competitive offers received
""")
