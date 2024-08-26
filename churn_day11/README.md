## Customer Churn Prediction Project
A machine learning project for predicting whether a customer will churn or not.

The dataset contains information like personal details, transaction history, and bank information of customers. Predicting when customers might withdraw their money and become inactive is crucial for banks. By identifying potential churners, the bank can take necessary actions to retain customers and maintain loyalty.

### Exploratory Data Analysis (EDA):

1. Categorical Columns: Analyzed churn distribution by categories such as gender, occupation, and customer network category using proportional aggregation.
2. Numerical Columns: Visualized the relationship between churn and numerical variables such as current balance, previous month balances, and transaction history using boxplots.
   
### Feature Engineering:
1. Extracted day, month, and day of the week from the last_transaction date column.
2. Removed outliers using the 3-sigma rule for various numeric variables to improve model performance.

### Model Building:
1. The data was split into training and testing sets with an 80/20 ratio.
2. Different machine learning models were tested, including Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM).
