import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
df = pd.read_csv(r'F:\CSL_Project\Mall_Customers.csv')  # Absolute path

# Step 2: Data Preprocessing
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Encode Gender
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male=0, Female=1

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
X = df[features]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: EDA
# Distribution of Spending Score
plt.figure(figsize=(10, 6))
sns.histplot(df['Spending Score (1-100)'], bins=30, kde=True)
plt.title('Distribution of Spending Score (1-100)')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.savefig(r'F:\CSL_Project\spending_score_distribution.png')
plt.close()

# Scatter plot: Annual Income vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', palette='Set1')
plt.title('Annual Income vs Spending Score by Gender')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig(r'F:\CSL_Project\income_vs_spending.png')
plt.close()

# Step 4: Determine Optimal Number of Clusters (Elbow Method)
inertia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.savefig(r'F:\CSL_Project\elbow_plot.png')
plt.close()

# Step 5: Apply K-Means Clustering (k=5 based on Elbow Method)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Evaluate Clusters
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.3f}')

# Cluster summary
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Summary:\n", cluster_summary)

# Step 7: Visualize Clusters
# Scatter plot: Annual Income vs Spending Score by Cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.title('Customer Segments: Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig(r'F:\CSL_Project\cluster_scatter.png')
plt.close()

# Additional Visualization 1: Box Plot of Features by Cluster (Fixed)
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
sns.boxplot(x='Cluster', y='Age', hue='Cluster', data=df, palette='Set2', legend=False)
plt.title('Age Distribution by Cluster')

plt.subplot(1, 3, 2)
sns.boxplot(x='Cluster', y='Annual Income (k$)', hue='Cluster', data=df, palette='Set2', legend=False)
plt.title('Annual Income Distribution by Cluster')

plt.subplot(1, 3, 3)
sns.boxplot(x='Cluster', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2', legend=False)
plt.title('Spending Score Distribution by Cluster')

plt.tight_layout()
plt.savefig(r'F:\CSL_Project\boxplot_features_by_cluster.png')
plt.close()

# Additional Visualization 2: Bar Plot of Gender Distribution per Cluster
gender_counts = df.groupby(['Cluster', 'Gender']).size().unstack()
gender_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#FF9999', '#66B2FF'])
plt.title('Gender Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.legend(['Female (1)', 'Male (0)'], title='Gender')
plt.savefig(r'F:\CSL_Project\gender_distribution_by_cluster.png')
plt.close()

# Additional Visualization 3: Pair Plot of Features Colored by Cluster
sns.pairplot(df, vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], hue='Cluster', palette='Set1', diag_kind='kde')
plt.suptitle('Pairwise Relationships of Features by Cluster', y=1.02)
plt.savefig(r'F:\CSL_Project\pairplot_features_by_cluster.png')
plt.close()

# Additional Visualization 4: Heatmap of Cluster Means
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Heatmap of Average Feature Values per Cluster')
plt.ylabel('Cluster')
plt.savefig(r'F:\CSL_Project\heatmap_cluster_means.png')
plt.close()

# Step 8: Save the Model
import joblib
joblib.dump(kmeans, r'F:\CSL_Project\kmeans_model_mall.pkl')
joblib.dump(scaler, r'F:\CSL_Project\scaler_mall.pkl')

# Step 9: Save Cluster Summary for Marketing
cluster_summary.to_csv(r'F:\CSL_Project\cluster_summary.csv')