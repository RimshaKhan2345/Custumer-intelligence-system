import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

st.title("🛍️ Customer Intelligence & Recommendation System")
st.markdown("Unsupervised Learning for E-commerce Analytics")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

@st.cache_data
def preprocess_data(df):
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, features

df = load_data()
X_scaled, scaler, features = preprocess_data(df)

# Sidebar navigation
task = st.sidebar.selectbox(
    "Select Task",
    ["Overview", "K-Means Clustering", "Anomaly Detection", 
     "PCA Visualization", "Recommendation System"]
)

if task == "Overview":
    st.header("📊 Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.write(f"- Total Customers: {len(df)}")
    st.write(f"- Features: {', '.join(df.columns)}")
    
    st.write("### Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feature in enumerate(features):
        axes[i].hist(df[feature], bins=20, edgecolor='black')
        axes[i].set_title(feature)
    st.pyplot(fig)

elif task == "K-Means Clustering":
    st.header("🎯 Customer Segmentation using K-Means")
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Cluster interpretation
    cluster_summary = df.groupby('Cluster')[features].mean()
    st.write("### Cluster Characteristics")
    st.dataframe(cluster_summary)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=clusters, cmap='viridis', alpha=0.6)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f'Customer Segments (k={n_clusters})')
    st.pyplot(fig)

elif task == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection using Gaussian Distribution")
    
    threshold = st.slider("Anomaly Threshold Percentile", 90, 99, 95)
    
    # Anomaly detection
    mean = np.mean(X_scaled, axis=0)
    cov = np.cov(X_scaled.T) + np.eye(3) * 1e-6
    mv_normal = multivariate_normal(mean=mean, cov=cov)
    densities = mv_normal.pdf(X_scaled)
    
    threshold_value = np.percentile(densities, threshold)
    anomalies = densities < threshold_value
    
    st.write(f"### Anomaly Detection Results")
    st.write(f"- Total Customers: {len(df)}")
    st.write(f"- Anomalies Found: {anomalies.sum()}")
    st.write(f"- Anomaly Percentage: {anomalies.mean()*100:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                    c=densities, cmap='RdYlGn_r', alpha=0.6)
    axes[0].set_title('Density Distribution')
    axes[0].set_xlabel(features[0])
    axes[0].set_ylabel(features[1])
    
    normal_idx = ~anomalies
    axes[1].scatter(df.loc[normal_idx, 'Annual Income (k$)'], 
                    df.loc[normal_idx, 'Spending Score (1-100)'],
                    c='blue', label='Normal', alpha=0.5)
    axes[1].scatter(df.loc[anomalies, 'Annual Income (k$)'], 
                    df.loc[anomalies, 'Spending Score (1-100)'],
                    c='red', label='Anomaly', s=100, marker='x')
    axes[1].set_xlabel('Annual Income (k$)')
    axes[1].set_ylabel('Spending Score')
    axes[1].set_title('Anomaly Visualization')
    axes[1].legend()
    
    st.pyplot(fig)

elif task == "PCA Visualization":
    st.header("📉 Dimensionality Reduction with PCA")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    st.write("### Variance Explained")
    explained_variance = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots()
    ax.bar(['PC1', 'PC2'], explained_variance)
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Component Variance')
    st.pyplot(fig)
    
    st.write(f"- PC1 captures {explained_variance[0]*100:.1f}% of variance")
    st.write(f"- PC2 captures {explained_variance[1]*100:.1f}% of variance")
    st.write(f"- Total variance preserved: {(explained_variance.sum())*100:.1f}%")
    
    # 2D PCA visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='blue')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Data in 2D PCA Space')
    st.pyplot(fig)

elif task == "Recommendation System":
    st.header("🎁 Collaborative Filtering Recommendations")
    
    # Create simulated ratings matrix
    n_products = 10
    ratings_matrix = np.zeros((len(df), n_products))
    
    for i in range(len(df)):
        base_rating = df.iloc[i]['Spending Score (1-100)'] / 20
        for j in range(n_products):
            rating = base_rating * (0.5 + 0.5 * np.sin(j * 0.5))
            rating = min(5, max(1, rating + np.random.normal(0, 0.3)))
            ratings_matrix[i, j] = rating
    
    ratings_df = pd.DataFrame(ratings_matrix, columns=[f'P{i+1}' for i in range(n_products)])
    
    # User selection
    user_id = st.selectbox("Select Customer ID", df.index[:100])
    
    # Collaborative filtering
    user_similarity = cosine_similarity(ratings_df)
    similar_users = user_similarity[user_id].argsort()[-6:-1][::-1]
    
    # Get recommendations
    user_ratings = ratings_df.loc[user_id].values
    weighted_ratings = np.zeros(n_products)
    total_sim = 0
    
    for sim_user in similar_users:
        sim = user_similarity[user_id][sim_user]
        weighted_ratings += sim * ratings_df.loc[sim_user].values
        total_sim += sim
    
    predicted_ratings = weighted_ratings / total_sim if total_sim > 0 else ratings_df.mean().values
    
    # Recommend products not already highly rated
    candidate_mask = user_ratings < 4
    recommendations = [(i, predicted_ratings[i]) for i in range(n_products) if candidate_mask[i]]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    st.write("### Customer Profile")
    st.write(f"- Age: {df.iloc[user_id]['Age']}")
    st.write(f"- Gender: {'Female' if df.iloc[user_id]['Gender']==1 else 'Male'}")
    st.write(f"- Annual Income: ${df.iloc[user_id]['Annual Income (k$)']}k")
    st.write(f"- Spending Score: {df.iloc[user_id]['Spending Score (1-100)']}/100")
    
    st.write("### Top Recommendations")
    for idx, (prod, score) in enumerate(recommendations[:5]):
        st.write(f"{idx+1}. **Product {prod+1}** - Predicted Rating: {score:.2f}/5")
    
    st.write("### How it works")
    st.info("This recommendation system finds customers similar to you and suggests products they liked!")