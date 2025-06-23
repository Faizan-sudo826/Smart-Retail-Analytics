🛒 Smart Retail Analytics – Data Mining Project in Python
A comprehensive data mining project on real-world retail transaction data using Python. This project explores customer behavior and purchasing patterns, applies unsupervised and supervised machine learning techniques, and extracts actionable business insights from the UCI Online Retail dataset.

📁 Dataset
Source: UCI Machine Learning Repository

Alternative Download: Kaggle – Ecommerce Data

Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

🚀 Project Overview
This project covers the following major steps in a typical data mining pipeline:

Data Cleaning & Preprocessing

Similarity and Dissimilarity Matrices

Customer Clustering

Association Rule Mining

Customer Value Prediction using Naïve Bayes

Support Vector Machine (SVM) Modeling

🔧 Technologies & Libraries
Python 3.x

pandas, numpy

scikit-learn

seaborn, matplotlib

mlxtend

Jupyter Notebook

✅ Project Tasks & Deliverables
1. 📊 Data Ingestion & Preprocessing
Loaded the dataset into a pandas DataFrame.

Cleaned missing CustomerID values.

Removed cancellations (InvoiceNo starting with 'C').

Engineered new features:

TotalAmount = Quantity × UnitPrice

RFM metrics: Recency, Frequency, Monetary

One-hot encoded the Country field.

✅ Deliverable: Cleaned and structured DataFrame ready for analysis.

2. 🧮 Similarity & Dissimilarity Matrices
Calculated Euclidean distance between customers using RFM features.

Calculated Jaccard similarity between sets of top 10 purchased products per customer.

Visualized matrices using seaborn.heatmap.

✅ Deliverable: Annotated Jupyter notebook with visualizations and commentary.

3. 🔍 Customer Clustering
Applied K-Means Clustering (with optimal K via elbow method/silhouette score).

Applied DBSCAN and tuned eps and min_samples.

Visualized clusters using PCA (2D scatter plots).

✅ Deliverable: Cluster comparison + plots + interpretation of noise points and groupings.

4. 🔗 Association Rule Mining
Applied Apriori and FP-Growth algorithms using mlxtend.

Mined rules with:

min_support = 0.02

confidence ≥ 0.6

lift ≥ 1.2

✅ Deliverable: Top 10 rules sorted by lift, along with runtime comparison.

5. 🧠 Naïve Bayes Classification
Task: Predict if a customer is High Value (Monetary > 75th percentile).

Models:

GaussianNB using RFM features.

BernoulliNB using binary purchase flags.

Visualized Confusion Matrices and ROC Curves.

✅ Deliverable: Model comparison with evaluation metrics.

6. 💻 Support Vector Machine (SVM)
Reused High-Value prediction task.

Models:

SVC(kernel='linear')

SVC(kernel='rbf')

Tuned hyperparameters (C, gamma) using GridSearchCV.

Visualized decision boundaries on PCA-reduced data.

✅ Deliverable: Accuracy summary + decision boundary plots.

📈 Key Outcomes
Identified distinct customer segments and high-value clients.

Discovered strong product purchase associations for marketing insights.

Built predictive models with solid classification accuracy.

Gained hands-on experience with real-world data and machine learning workflows.

🔄 How to Run
Clone this repo:

bash
Copy
Edit
git clone https://github.com/your-username/smart-retail-analytics.git
cd smart-retail-analytics
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook SmartRetailAnalytics.ipynb
📚 Learnings & Skills Applied
End-to-end data mining process using Python

Feature engineering for customer analytics

Clustering and similarity measures

Market basket analysis (Apriori, FP-Growth)

Classification using Naïve Bayes and SVM

Data visualization and interpretability
