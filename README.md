# Dan_Data_Science_Portfolio

## [Classification Project (Kaggle):  "Titanic - Machine Learning from Disaster"](https://github.com/drossDS/Project-Classification-Titanic_Machine_Learning)
Data for passengers on the Titanic is provided and a machine learning classification model is written to predict the survival of a subset of the passengers
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering<br><br>
![](/Images/Correlation_Matrix_small.png)<br>
- Employed hypothesis testing validate the statistical significance of engineered features<br><br>
![](/Images/Age_Distro_Swarm_small.png)
![](/Images/Survival_Ratio_vs_Cumulative_Age_Group.png)<br>
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used GridSearchCV to optimize models to increase model accuracy
- ShuffleSplit was employed to generate training and validation sets reduce overfitting by simulating the effects of unseen data.  (Below, the "Test" column is actually the average performance for each model against the generated validation sets)<br><br>
![](/Images/Model_Comparison_Table.png)
