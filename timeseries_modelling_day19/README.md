In this project, I performed analysis on one AirPassengersDataset selected from the Darts collection. I carried out Exploratory Data Analysis (EDA) to check the trend and seasonality of the dataset used. We include interpretations of trend and seasonality charts as well as decomposition of timeseries data.

After that, I tested whether the dataset used was included in the additive or multiplicative category using the Augmented Dickey-Fuller (ADF) test, and concluded the results.

Then, I applied several models, namely, Auto ARIMA, Naive Model Seasonal, and Drift. I also use Prophet to make predictions and evaluate the performance of the models.

From the results above, the Naive Drift Naive Seasonal model has a higher Rsquare value compared to the other 2 models and smaller MAPE and RMSE values than the ARIMA and Prophet models.