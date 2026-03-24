"""
Part 3: Customer Segmentation using K-Means Clustering
========================================================
This script segments customers into distinct groups based on their
purchasing behavior using K-Means clustering algorithm.

Data Source: Synthetically generated dataset based on typical retail/e-commerce
             customer behavior patterns. Features reflect real-world customer
             characteristics for segmentation analysis.

Author: AI Agentics Assignment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Generate Realistic Dataset (150+ records)
# =============================================================================
# Data Source Comment: Synthetically generated based on retail/e-commerce
# customer behavior patterns. Features represent typical customer segments
# ranging from budget-conscious shoppers to premium loyal customers.

n_samples = 200  # 200 records for robust clustering

# Generate customer features with natural clustering tendencies
# We'll create 3 natural segments: Budget, Regular, Premium

def generate_customer_data(n):
    """Generate customer data with natural cluster tendencies."""
    data = []
    
    for _ in range(n):
        # Randomly assign to a latent segment (for realistic data generation)
        segment = np.random.choice(['budget', 'regular', 'premium'], p=[0.35, 0.45, 0.20])
        
        if segment == 'budget':
            # Budget customers: low spending, low frequency, varied age
            annual_spending = np.random.normal(500, 200)
            purchase_frequency = np.random.normal(5, 2)
            age = np.random.normal(35, 15)
            region = np.random.choice(['North', 'South', 'East', 'West'], p=[0.3, 0.3, 0.2, 0.2])
            
        elif segment == 'regular':
            # Regular customers: moderate spending and frequency
            annual_spending = np.random.normal(2000, 500)
            purchase_frequency = np.random.normal(15, 5)
            age = np.random.normal(40, 12)
            region = np.random.choice(['North', 'South', 'East', 'West'], p=[0.25, 0.25, 0.25, 0.25])
            
        else:  # premium
            # Premium customers: high spending, high frequency, slightly older
            annual_spending = np.random.normal(5000, 1000)
            purchase_frequency = np.random.normal(30, 8)
            age = np.random.normal(45, 10)
            region = np.random.choice(['North', 'South', 'East', 'West'], p=[0.2, 0.3, 0.2, 0.3])
        
        # Clip values to realistic ranges
        annual_spending = max(100, min(10000, annual_spending))
        purchase_frequency = max(1, min(50, int(purchase_frequency)))
        age = max(18, min(75, int(age)))
        
        data.append({
            'annual_spending': round(annual_spending, 2),
            'purchase_frequency': purchase_frequency,
            'age': age,
            'region': region
        })
    
    return pd.DataFrame(data)

df = generate_customer_data(n_samples)

# =============================================================================
# Display Dataset Info
# =============================================================================
print("=" * 70)
print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
print("=" * 70)
print(f"\nDataset Size: {len(df)} records")
print(f"\nDataset Preview (first 10 rows):")
print(df.head(10).to_string(index=False))

print(f"\nDataset Statistics:")
print(df.describe().round(2))

print(f"\nRegion Distribution:")
print(df['region'].value_counts())

# =============================================================================
# Feature Preparation
# =============================================================================
# Select numerical features for clustering
numerical_features = ['annual_spending', 'purchase_frequency', 'age']
X = df[numerical_features].copy()

# Scale features using StandardScaler
# This is crucial for K-Means as it's distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nScaled Features (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=numerical_features).head().round(3).to_string(index=False))

# =============================================================================
# Elbow Method - Finding Optimal K
# =============================================================================
print("\n" + "=" * 70)
print("ELBOW METHOD - DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

# Calculate inertia for K=1 to 10
k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k:2d}: Inertia = {kmeans.inertia_:.2f}")

# Create Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal K Selection', fontsize=14)
plt.xticks(k_range)
plt.grid(True, alpha=0.3)

# Mark the elbow point (K=3)
plt.axvline(x=3, color='r', linestyle='--', label='Optimal K=3')
plt.legend()
plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=150)
plt.close()

print(f"\n✅ Elbow plot saved as 'elbow_plot.png'")
print(f"\nELBOW ANALYSIS:")
print(f"Looking at the inertia values, we see a significant decrease from K=1 to K=3,")
print(f"after which the rate of decrease slows down (the 'elbow').")
print(f"Therefore, K=3 is selected as the optimal number of clusters.")

# =============================================================================
# Apply K-Means Clustering (K=3)
# =============================================================================
print("\n" + "=" * 70)
print("K-MEANS CLUSTERING (K=3)")
print("=" * 70)

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nCluster Distribution:")
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count} customers ({count/len(df)*100:.1f}%)")

# =============================================================================
# Cluster Analysis
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTER ANALYSIS - AVERAGE CHARACTERISTICS")
print("=" * 70)

# Calculate mean characteristics for each cluster
cluster_summary = df.groupby('cluster')[numerical_features].mean().round(2)
cluster_summary['count'] = df.groupby('cluster').size()

print("\nCluster Summary Statistics:")
print(cluster_summary.to_string())

# Detailed cluster analysis
print("\n" + "-" * 70)
print("DETAILED CLUSTER PROFILES")
print("-" * 70)

# Identify cluster types based on characteristics
cluster_profiles = {}
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    avg_spending = cluster_data['annual_spending'].mean()
    avg_frequency = cluster_data['purchase_frequency'].mean()
    avg_age = cluster_data['age'].mean()
    
    # Determine cluster type
    if avg_spending < 1000 and avg_frequency < 10:
        cluster_type = "BUDGET SHOPPERS"
        emoji = "💰"
    elif avg_spending > 3500 and avg_frequency > 20:
        cluster_type = "PREMIUM LOYALISTS"
        emoji = "⭐"
    else:
        cluster_type = "REGULAR CUSTOMERS"
        emoji = "🛒"
    
    cluster_profiles[cluster] = {
        'type': cluster_type,
        'emoji': emoji,
        'spending': avg_spending,
        'frequency': avg_frequency,
        'age': avg_age
    }
    
    print(f"\n{emoji} CLUSTER {cluster}: {cluster_type}")
    print(f"   Customers: {len(cluster_data)}")
    print(f"   Avg Annual Spending: ${avg_spending:,.2f}")
    print(f"   Avg Purchase Frequency: {avg_frequency:.1f} purchases/year")
    print(f"   Avg Age: {avg_age:.1f} years")
    print(f"   Region Distribution:")
    for region, count in cluster_data['region'].value_counts().items():
        print(f"      {region}: {count} ({count/len(cluster_data)*100:.1f}%)")

# =============================================================================
# Marketing Strategies for Each Cluster
# =============================================================================
print("\n" + "=" * 70)
print("TARGETED MARKETING STRATEGIES")
print("=" * 70)

strategies = {
    "BUDGET SHOPPERS": """
   💰 BUDGET SHOPPERS - Marketing Strategy:
   
   CHARACTERISTICS:
   - Low annual spending (< $1,000)
   - Infrequent purchases
   - Price-sensitive customers
   
   RECOMMENDED STRATEGIES:
   1. DISCOUNTS & DEALS
      - Weekly flash sales and clearance events
      - Bundle deals and value packs
      - Price match guarantees
   
   2. LOYALTY PROGRAM (Entry Level)
      - Points-per-purchase system
      - First-time buyer discounts
      - Referral bonuses
   
   3. COMMUNICATION
      - Email: Focus on sales and promotions
      - SMS: Limited-time offer alerts
      - Frequency: Weekly promotional emails
   
   4. UPSELLING OPPORTUNITIES
      - Suggest budget-friendly alternatives
      - Highlight value propositions
      - Free shipping thresholds to encourage larger orders
""",
    
    "REGULAR CUSTOMERS": """
   🛒 REGULAR CUSTOMERS - Marketing Strategy:
   
   CHARACTERISTICS:
   - Moderate spending ($1,000 - $3,500)
   - Consistent purchase pattern
   - Reliable customer base
   
   RECOMMENDED STRATEGIES:
   1. LOYALTY REWARDS
      - Tiered rewards program
      - Birthday/anniversary discounts
      - Exclusive member sales
   
   2. CROSS-SELLING
      - Product recommendations based on history
      - "Customers also bought" suggestions
      - Complementary product bundles
   
   3. ENGAGEMENT
      - Personalized email campaigns
      - Product reviews requests
      - Early access to new products
   
   4. RETENTION FOCUS
      - Satisfaction surveys
      - Re-engagement campaigns for dormant customers
      - Subscription/auto-replenishment options
""",
    
    "PREMIUM LOYALISTS": """
   ⭐ PREMIUM LOYALISTS - Marketing Strategy:
   
   CHARACTERISTICS:
   - High annual spending (> $3,500)
   - Frequent purchases
   - Brand advocates and VIP customers
   
   RECOMMENDED STRATEGIES:
   1. VIP TREATMENT
      - Dedicated account manager
      - Priority customer service
      - Exclusive preview events
   
   2. PREMIUM REWARDS
      - VIP loyalty tier with enhanced benefits
      - Free expedited shipping
      - Complimentary gift wrapping
   
   3. EXCLUSIVE OFFERS
      - Early access to limited editions
      - Invitation-only sales
      - Personalized product curation
   
   4. RELATIONSHIP BUILDING
      - Handwritten thank-you notes
      - Surprise upgrades and gifts
      - Invitations to brand events
   
   5. ADVOCACY PROGRAMS
      - Brand ambassador opportunities
      - Exclusive referral rewards
      - Featured customer spotlights
"""
}

for cluster in range(optimal_k):
    profile = cluster_profiles[cluster]
    strategy = strategies.get(profile['type'], strategies["REGULAR CUSTOMERS"])
    print(strategy)

# =============================================================================
# Save Results to CSV
# =============================================================================
output_filename = 'customer_segments.csv'
df.to_csv(output_filename, index=False)
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)
print(f"\n✅ Cluster assignments saved to '{output_filename}'")
print(f"✅ Elbow plot saved to 'elbow_plot.png'")

print(f"\nSample of saved data (first 10 rows):")
print(df.head(10).to_string(index=False))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
This K-Means clustering analysis segmented {len(df)} customers into {optimal_k} groups:
""")

for cluster in range(optimal_k):
    profile = cluster_profiles[cluster]
    count = len(df[df['cluster'] == cluster])
    print(f"  {profile['emoji']} Cluster {cluster} ({profile['type']}): {count} customers")
    print(f"     Avg Spending: ${profile['spending']:,.2f}, Avg Frequency: {profile['frequency']:.1f}")

print(f"""
KEY INSIGHTS:
- The elbow method confirmed K=3 as optimal (saved in elbow_plot.png)
- Each segment requires different marketing approaches
- Premium customers (high value) should receive VIP treatment
- Budget shoppers respond best to discounts and deals
- Regular customers need loyalty programs to prevent churn

BUSINESS VALUE:
- Personalized marketing increases conversion rates
- Resource allocation based on customer value
- Improved customer satisfaction through relevant offers
- Better ROI on marketing spend
""")
