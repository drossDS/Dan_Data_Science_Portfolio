<!-- keep this here -->

# Contents:
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


# Titanic Classification Model 
## Kaggle Project:  "Titanic - Machine Learning from Disaster"
Link to code [here](https://github.com/drossDS/Project-Classification-Titanic_Machine_Learning)


# [Classification Project (Kaggle):  "Titanic - Machine Learning from Disaster"](https://github.com/drossDS/Project-Classification-Titanic_Machine_Learning)
A classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Training and test data sets were provided by Kaggle.  The training data set was provided with passenger outcomes to train and fit various classification/machine learning algorithms.  The test data was provided without the passenger outcomes, and the developed model was used to predict their fates.<br>
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
