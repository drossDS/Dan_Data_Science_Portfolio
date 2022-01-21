<!-- keep this here -->

# Project Portfolio Contents:
- Titanic Classification Model (Self Directed)
- Carbon Emissons Timeseries Forecasting (MIT ADSP Capstone Project)
- Unsupervised Learning (Clustering) Project (MIT ADSP)
- Supervised Learning (Regression Analsysis) Project (MIT ADSP)
- Linear Regression / Machine Learning Project (MIT ADSP)
- Deep Learning (Neural Network) Project (MIT ADSP)
- Recommendation System Project (MIT ADSP)
- ***Coming Soon!***
  - SQL Project
  - Self-Directed Clustering Project
<!--
h1{
    margin-top: 0px;
}
-->

# Titanic Classification Model 
***Kaggle Competition:  "Titanic - Machine Learning from Disaster"*** - link to python code [here](https://github.com/drossDS/Project-Classification-Titanic_Machine_Learning)


<!--# [Classification Project (Kaggle):  "Titanic - Machine Learning from Disaster"](https://github.com/drossDS/Project-Classification-Titanic_Machine_Learning)-->
A classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Data sets were provided by Kaggle and included a training data set with passenger outcomes model fitting and a test data set on which to run the model and submit the results for the competition.<br>
### Major Accomplishments:
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering<br><br>
![](/Images/Classification_Titanic/Correlation_Matrix_small.png)<br><br>

- Employed hypothesis testing validate the statistical significance of engineered features<br><br>
![](/Images/Classification_Titanic/Age_Distro_Swarm_small.png)
![](/Images/Classification_Titanic/Survival_Ratio_vs_Cumulative_Age_Group.png)<br><br>
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used sklearn GridSearchCV to optimize models to increase model accuracy
- Generated training and validation sets using sklearn ShuffleSplit to simulate the effects of unseen data and reduce overfitting<br><br>

![](Images/Classification_Titanic/Model_Comparison_Table.png)<br>

### Key Outcomes:
- A random forest classifier model was chosen with a predicted accuracy of about 82% based on validation data
- The chosen random forest model predicted the test data with a 77.3% accuracy
  - Next steps and model refinements are proposed in the code to improve this closer to the 82% prediction accuracy achieved on the validation data sets

# Carbon Emissons Timeseries Forecasting
***Capstone Project - MIT Applied Data Science Program***

A regression model was created to forecast future carbon emissions using time series data for electric energy production emissions from natural gas in the US from 1973 to 2016.

![](Images/Regression_Emissions/Provided_NNEIEUS_Data.png)
### Major Accomplishments:
- A model was created to forecast future emissions for a period of 1 year beyond the provided data
- Training Data Processing:
  - Different data transformation techniques were evaluated to optimize model performance
  - Various training data timespans over which the model would be fit were also examined to reduce prediction error
- Model Development:
  - Basic AR, MA, ARMA, and ARIMA models were optimized and evaluated
  - A Seasonal ARIMA (SARIMA) model was tuned using the Pmdarima auto_arima optimizer function
- Model performance was primarily characterized by calculating root mean squared error (RMSE) values for each combination of model, transformation technique, and fit data timespan<br>

### Key Outcomes:
- A 3% mean absolute percent error (MAPE) for the final model was calculated against the validation data set<br>
![](Images/Regression_Emissions/Carbon_Emissions_Validation.png)<br>
- Carbon emissions levels were forecasted for the 12 months following the provided data
![](Images/Regression_Emissions/Carbon_Emissions_Forecast.png)

