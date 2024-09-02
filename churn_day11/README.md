# Bank Customer Churn Prediction
This project aims to predict customer churn using various machine learning models. The dataset contains customer-related information and is used to build a predictive model to identify customers likely to churn.

## Table of Contents
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-EDA)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Excel Analysis](#Excel-Analysis)

## Dataset

The dataset used in this project is a Bank Customer Churn dataset. It contains various features related to customer transactions and behaviors, including the `churn` column, which indicates whether a customer has churned or not.

## Libraries Used

The following Python libraries are used in this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikit-plot`

## Exploratory Data Analysis (EDA)

1. Categorical Columns: Analyzed churn distribution by categories such as gender, occupation, and customer net Worth category using proportional aggregation.
2. Numerical Columns: Visualized the relationship between churn and numerical variables such as current month, previous month and previous quarter using boxplots.
   
## Data Preprocessing

### Steps:

1. **Loading Data**: The dataset is loaded using `pandas`.

   ```python
   data = pd.read_csv('churn_prediction.csv')
   data.head()

2. **Handling Missing Values**: Rows with missing values in the last_transaction column are removed.

   ```python
   data = data[data['last_transaction'] != 'NaT'].reset_index(drop=True)
   data['last_transaction'] = pd.to_datetime(data['last_transaction'])

3. **Feature Engineering**:

   ```python
   date = pd.DatetimeIndex(data['last_transaction'])
   data['doy_ls_tran'] = date.dayofyear
   data['moy_ls_tran'] = date.month
   data['dow_ls_tran'] = date.dayofweek
   data = data.drop(columns=['last_transaction'])

4. **Encoding Categorical Variables**: One-hot encoding is applied to convert categorical variables into numerical form.

   ```python
   data = pd.get_dummies(data, drop_first=True, dtype=int)

5. **Splitting the Dataset**: The dataset is split into training and testing sets using an 80-20 split.

   ```python
   train, test = train_test_split(data, test_size=0.2, random_state=42)
   X_train = train.drop(columns=['churn'])
   y_train = train['churn']
   X_test = test.drop(columns=['churn'])
   y_test = test['churn']

## Modeling
The following machine learning models were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors Classifier (KNN)
- Decision Tree Classifier
- Support Vector Classifier (SVC)

## Evaluation
The model was evaluated on both training and test data using metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score

Additional analysis included plotting cumulative gain and lift curves.
1. **Cumulative Gain Chart**
   ```python
   skplt.metrics.plot_cumulative_gain(y_test, logistic_model.predict_proba(X_test))
   plt.grid(True)
   plt.show()
3. **Lift Curve**
   ```python
   skplt.metrics.plot_lift_curve(y_test, logistic_model.predict_proba(X_test))

## Conclusion
- The Logistic Regression model achieved good in Recall and was able to identify a significant portion of customers likely to churn.
- From the cumulative gain chart, it was observed that taking a 45% sample results in capturing 75% of churn customers.

## Excel Analysis
The included Excel file, Profitable-Business-Using-Cumulative-Gain-Plot.xlsx. This Excel file is used to calculate and compare the profitability of using the predictive model versus a random sampling baseline.

### Profitability Formulae:
#### Saved:
- Saved refers to the number of people who were correctly identified as churn by the model or random sampling. This is calculated based on the actual number of churns that match the predictions.
#### Failed:
- Failed refers to two types of errors:
1. False Negatives: People predicted not to churn but who actually churn.
2. False Positives: People predicted to churn but who actually do not churn.
- Failed = ABS(y_test (churn, actual data) - y_pred (churn, predicted data))
#### Campaign Cost Assumption:
- Campaign Cost per Person: Rp 30.000 (approximately $2 per month).
#### Potential Profit per Person:
- Potential Profit per Person: 15% of the campaign cost.
- Potential Profit per Person = 15% * Campaign Cost
#### Failed Cost:
- Failed Cost represents the cost incurred from failed predictions, calculated by adding the cost of the campaign for each failed prediction and the lost potential profit.
- Failed Cost = (Failed * Campaign Cost) + (Failed * Potential Profit per Person)
#### Total Cost:
- Total Cost is the total amount spent on the campaign based on the number of people predicted to churn.
- Total Cost = Campaign Cost * Number of People Predicted to Churn
#### Profit:
- Profit is calculated by multiplying the number of people saved by the potential profit per person, adjusted by the total cost.
- Profit = (Saved * Potential Profit per Person) - Total Cost
#### Netto Profit:
- Netto Profit is the final profit after subtracting both the total cost and failed cost.
- Netto Profit = Profit - Total Cost - Failed Cost
