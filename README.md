
# <img alt="bizkit" src="logo.jpg" height="300" width="300">


# bizkit

bizkit is a Python package to help streamlining business analytics data mining tasks. This package provides models for market basket analysis, anomaly detection, time-to-event modelling, customer segmentation, and uplift modelling (in progress). Implemented algorithms include mlxtend.apriori, sklearn.IsolationForest, lifelines.KaplanMeierFitter, k-means, and catboost. bizkit focuses on ease of use by providing a well-documented and consistent interface. The results are presented in the interactive visualization via libraries of bokeh, d3fgraph, and plotly.

# Overview

**Market Basket Analysis:**
- Market basket analysis allows retailers to identify relationships between the items that customer purchase. This form of association analysis is much simpler to implement than many traditional types of ML (clustering, regression, Neural Networks, etc.) and the results are relatively easy to interpret. Association analysis will focus on identifying association rules using the MLxtend Library.

**Anomaly Detection:**
- Anomaly detection (unsupervised) helps e-commerce businesses to identify the busiest time of customer purchase and web browsing. Sklearn Isolation Forest, an ensemble regressor, is applied as the classifier for the identification. The results are plotted via bokeh into an interactive time series visualization with red dots representing anomalies.

**Time-to-Event Modelling:**
- Time-to-Event is a prediction of the net profit attributed to the entire future relationship with a customer. As it typically follows a time-to-event data structure, we can implement a survival model (specifically, the inverse of the model) to make inferences and predictions.

**Customer Segmentation:**
- Customer Segmentation is the activity of grouping data points, namely business customers, into clusters using unsupervised learning techniques. Analyzing the features of various clusters allows businesses to better understand their customers in the process of driving business growth. We implement a model that utilizes the KMeans algorithm to cluster customers into groups and produce visualizations such as radar plots, 3d scatter plots, feature distribution plots across clusters, and rug plots. To ease the process of tuning hyperparameter k, we introduce helper functions that use the elbow method. 


# Reference

Market Basket Analysis:
- Introduction to Market Basket Analysis in Python https://pbpython.com/market-basket-analysis.html
- A Gentle Introduction on Market Basket Analysis — Association Rules https://towardsdatascience.com/a-gentle-introduction-on-market-basket-analysis-association-rules-fa4b986a40ce



Anomaly detection:
- Anomaly Detection Principles and Algorithms (2017) by Kishan G. Mehrotra, Chilukuri K. Mohan, HuaMing Huang
- Sklearn IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html


Time-to-event modelling:
- Lifelines https://lifelines.readthedocs.io/en/latest/


Customer segmentation:
- Understanding K-means Clustering in Machine Learning https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1


Uplift Modelling (in progress):
- Simple Machine Learning Techniques To Improve Your Marketing Strategy: Demystifying Uplift Models https://medium.com/datadriveninvestor/simple-machine-learning-techniques-to-improve-your-marketing-strategy-demystifying-uplift-models-dc4fb3f927a2


# Author
Bassim Eledath, Lynn He, Christine Zhu, Amanda Ma
