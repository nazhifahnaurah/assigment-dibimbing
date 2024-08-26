## Introduction 
This project focuses on the Iris dataset, performing data analysis, clustering, and multiclass classification. The goal is to identify patterns in the data and use a machine learning model to classify iris species based on their features. The dataset includes the following features, according to its official summary: 
1. Petal Length (cm)
2. Petal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)
5. Species = setosa, versicolor, virginica

## Exploratory Data Analysis (EDA) 
**Visualization**: Several plots were created to explore the relationship between the features (petal length, petal width, petal length, and petal width) and the corresponding species (Setosa, Versicolor, Virginica).
**Insights**: 
- The Setosa species has a wider petal width and shorter petal length compared to the other species. 
- The Versicolor species shows a slight increase in both petal length and width.
- Virginica species has the longest and widest petals.
- Setosa is well separated from the other two species based on petal and corolla features.

## Feature Engineering 
 **Outlier Handling**: Addressing potential outliers in the dataset.
 **Feature Scaling**: StandardScaler is applied to normalize features for better performance in clustering and classification models.

## Clustering using KMeans 
**Elbow Method**: The optimal number of clusters is determined using the elbow method, which shows a clear separation between clusters.
**Score Plot**: The score plot further validates the optimal clustering, which shows a better distribution of clusters compared to annotator labels.
**Model Evaluation**: Both the elbow method and score plot show successful identification of data patterns with clear separation between clusters.

## Multiclass Classification 
- Models Used: Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.
**Training**: All features were used with the `cluster_elbow` column as the target variable for classification.
**Evaluation**: The `species` column was used to compare annotator labels with the predicted results.
**Results**: - All models performed well, with high accuracy on both training and testing data.
- Logistic Regression was selected as the best model.

## Model Evaluation 
Despite slight mixing in species labels, the machine learning models performed well in predicting the correct class. Logistic Regression proved to be a good model for multiclass classification.
