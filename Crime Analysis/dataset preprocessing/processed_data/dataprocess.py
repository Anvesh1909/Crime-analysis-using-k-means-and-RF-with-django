# Crime Data Analysis and Prediction
# Using Random Forest Regressor and K-means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the crime data
def load_data():
    # For demonstration, creating a dataframe from the provided sample
    data = {
        'States/UTs': ['ANDHRA PRADESH', 'ANDHRA PRADESH', 'ANDHRA PRADESH', 'ANDHRA PRADESH', 'ANDHRA PRADESH'],
        'District': ['ADILABAD', 'ANANTAPUR', 'CHITTOOR', 'CUDDAPAH', 'EAST GODAVARI'],
        'Year': [2001, 2001, 2001, 2001, 2001],
        'Murder': [101, 151, 101, 80, 82],
        'RAPE': [50, 23, 27, 20, 23],
        'THEFT': [199, 366, 723, 173, 1021]
    }
    return pd.DataFrame(data)

# Load and clean the data
df = load_data()
print("Dataset Overview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data preprocessing
# In a real scenario, we would need more data points for effective modeling
print("\nData Preprocessing...")

# Feature engineering: Create a total crime column
df['Total_Crime'] = df['Murder'] + df['RAPE'] + df['THEFT']
print("\nAdded Total Crime Feature:")
print(df[['States/UTs', 'District', 'Year', 'Total_Crime']].head())

# Visualization of crime distribution
plt.figure(figsize=(15, 10))

# Visualize crime types distribution by district
plt.subplot(2, 2, 1)
crime_data = df.melt(id_vars=['District'], value_vars=['Murder', 'RAPE', 'THEFT'], var_name='Crime_Type', value_name='Count')
sns.barplot(x='District', y='Count', hue='Crime_Type', data=crime_data)
plt.title('Crime Type Distribution by District')
plt.xticks(rotation=45)
plt.tight_layout()

# Visualize total crime by district
plt.subplot(2, 2, 2)
sns.barplot(x='District', y='Total_Crime', data=df)
plt.title('Total Crime by District')
plt.xticks(rotation=45)
plt.tight_layout()

# Correlation heatmap
plt.subplot(2, 2, 3)
crime_cols = ['Murder', 'RAPE', 'THEFT', 'Total_Crime']
correlation = df[crime_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Between Crime Types')
plt.tight_layout()

# Pie chart of crime types
plt.subplot(2, 2, 4)
crime_sums = [df['Murder'].sum(), df['RAPE'].sum(), df['THEFT'].sum()]
plt.pie(crime_sums, labels=['Murder', 'RAPE', 'THEFT'], autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Different Crime Types')
plt.tight_layout()

plt.savefig('crime_visualization.png')
plt.close()

print("\nVisualizations created and saved to 'crime_visualization.png'")

# K-means Clustering Analysis
print("\nPerforming K-means Clustering...")

# Prepare data for clustering
X_cluster = df[['Murder', 'RAPE', 'THEFT']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, min(6, len(df) + 1))  # Adjusted to handle small sample size
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
plt.close()

# Apply K-means with the optimal number of clusters
# For this small dataset, let's use 2 clusters
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nK-means clustering completed with {optimal_k} clusters")
print(df[['District', 'Murder', 'RAPE', 'THEFT', 'Cluster']].head())

# Visualize the clusters (using PCA for dimensionality reduction)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('Crime Clusters Visualization using PCA')
for i, txt in enumerate(df['District']):
    plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]))
plt.savefig('crime_clusters.png')
plt.close()

# Analyze cluster characteristics
cluster_analysis = df.groupby('Cluster')[['Murder', 'RAPE', 'THEFT']].mean()
print("\nCluster Characteristics:")
print(cluster_analysis)

# Create a plotting function for cluster characteristics
plt.figure(figsize=(12, 8))
cluster_analysis.T.plot(kind='bar')
plt.title('Average Crime Rates by Cluster')
plt.ylabel('Average Number of Cases')
plt.xlabel('Crime Type')
plt.legend(title='Cluster')
plt.savefig('cluster_characteristics.png')
plt.close()

# Predictive Modeling with Random Forest
print("\nTraining Random Forest Regressor for crime prediction...")

# Note: In reality, we would need more data points and time series data for meaningful prediction
# This is a demonstration with a very limited dataset

# For demonstration purposes, let's try to predict total crime from other features
X = df[['Murder', 'RAPE']]  # Using murder and rape to predict theft
y = df['THEFT']

# Since the dataset is small, we will use a simple train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Model evaluation
print("\nRandom Forest Model Evaluation:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Predicting THEFT')
plt.savefig('feature_importance.png')
plt.close()

# Future prediction simulation
print("\nSimulating Future Crime Prediction...")

# Create a hypothetical future scenario
# In reality, this would use time series forecasting methods with more historical data
new_data = pd.DataFrame({
    'Murder': [100, 120, 90, 85, 95],
    'RAPE': [30, 25, 28, 22, 24]
})

# Use the trained model to predict theft
predicted_theft = rf_model.predict(new_data)

# Create a dataframe with the predictions
future_predictions = pd.DataFrame({
    'Murder': new_data['Murder'],
    'RAPE': new_data['RAPE'],
    'Predicted_THEFT': predicted_theft
})

print("\nFuture Crime Predictions:")
print(future_predictions)

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.scatter(range(len(future_predictions)), future_predictions['Predicted_THEFT'], color='red', label='Predicted Theft')
plt.plot(range(len(future_predictions)), future_predictions['Predicted_THEFT'], color='red')
plt.title('Predicted THEFT Cases for Hypothetical Scenarios')
plt.xlabel('Scenario Number')
plt.ylabel('Number of THEFT Cases')
plt.legend()
plt.savefig('future_predictions.png')
plt.close()

# Conclusion
print("\nAnalysis Conclusion:")
print("1. The dataset provided contains crime statistics for districts in Andhra Pradesh for the year 2001.")
print("2. K-means clustering identified distinct crime patterns among the districts.")
print("3. The Random Forest model shows relationships between different crime types.")
print("4. For more accurate predictions, a larger dataset with temporal information would be needed.")
print("5. Feature importance analysis reveals which crime indicators are most predictive of others.")

print("\nNotes on improving the analysis with a more complete dataset:")
print("1. Include multiple years of data to analyze crime trends over time")
print("2. Include socioeconomic factors that might influence crime rates")
print("3. Use time series analysis for more accurate future predictions")
print("4. Consider spatial analysis to understand geographical patterns of crime")
print("5. Include more features for better clustering and predictive performance")