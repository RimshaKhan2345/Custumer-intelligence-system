# Custumer-intelligence-system
this is the unsupervised learning Custumer intelligence &amp; recommended system to detect annomalies in custumers behaviour.

# 🛍️ Unsupervised Learning for Customer Intelligence & Recommendations

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

## 📌 Project Overview

This project applies **unsupervised learning techniques** to analyze customer behavior for an e‑commerce platform. The goal is to uncover hidden patterns, detect anomalies, reduce dimensionality, and build a basic recommendation system – all without using labelled data.

The following methods are implemented:

- **K‑Means Clustering** – Customer segmentation  
- **Gaussian Density Estimation** – Anomaly detection  
- **Principal Component Analysis (PCA)** – Dimensionality reduction  
- **Collaborative Filtering** – Product recommendations  

A fully interactive **Streamlit dashboard** is provided to explore all tasks in real time.

---

## 🗂️ Dataset

We use the [**Mall Customers Dataset**](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle. It contains 200 customer records with the following features:

- `CustomerID` – Unique identifier  
- `Gender` – Male / Female  
- `Age` – Customer age (years)  
- `Annual Income (k$)` – Yearly income in thousand dollars  
- `Spending Score (1-100)` – Score assigned by the mall based on customer behaviour  

This dataset is clean, easy to understand, and perfectly suited for unsupervised learning tasks.

> 💡 *You can replace it with any other e‑commerce or customer dataset – just update the feature names in the code.*

---

## 🧩 Project Tasks

| Task | Description | Key Techniques |
|------|-------------|----------------|
| **1. Preprocessing** | Handle missing values, encode categorical variables, scale features | `StandardScaler`, `pandas` |
| **2. K‑Means Clustering** | Segment customers, select optimal k (elbow + silhouette), interpret clusters | `KMeans`, `silhouette_score` |
| **3. Anomaly Detection** | Detect unusual customers using multivariate Gaussian distribution | `scipy.stats.multivariate_normal` |
| **4. PCA** | Reduce dimensionality, visualise variance explained | `PCA` from scikit‑learn |
| **5. Recommendation System** | User‑based collaborative filtering with cosine similarity | `cosine_similarity` |
| **6. Streamlit Dashboard** | Interactive UI for all the above tasks | `streamlit` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/unsupervised-learning-project.git
cd unsupervised-learning-project
