<!-- keep this here -->

<div align="center">
    <center><h1>About Me</h1></center>
</div>
Hello!  I'm a mechanical engineer with over 11 years of experience aspiring to become a data scientist.  Just last year, I completed the MIT Applied Data Science Program (MIT ADSP) and have since  decided to pursue this field for my next career move.  Below are the projects from my course and the first of what I hope will be many more self-directed projects in areas of specific interest.  More to come soon!

<div align="center">
    <center><h1>Project Portfolio Contents</h1></center>
</div>
  
- Titanic Classification Model (Self-Directed)
- Carbon Emissions Timeseries Forecasting (MIT ADSP Capstone Project)
- Automobile Clustering - Unsupervised Learning Project (MIT ADSP)
- Boston Home Prices - Linear Regression Project (MIT ADSP)
- Image Recognition - Neural Networks Project (MIT ADSP)
- Contact Information

<!--
h1{
    margin-top: 0px;
}
-->

<!--                                      *********************************** 1 - TITANIC ************************************************* -->

---

<div align="center">
    <center><h1>Titanic Classification Model</h1></center>
</div>

<div align="center">
    <center><b><i>Self-Directed Project, Kaggle Competition:  "Titanic - Machine Learning from Disaster"</i></b></center>
</div>

A classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Datasets were provided by Kaggle and included a training dataset with passenger outcomes model fitting and a test dataset on which to run the model and submit the results for the competition.<br>
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
  - Next steps and model refinements are proposed in the code to improve this closer to the 82% prediction accuracy achieved on the validation datasets

<!--                                *********************************** 2 - CARBON EMISSIONS ************************************************* -->

---

<div align="center">
    <center><h1>Carbon Emissions Timeseries Forecasting</h1></center>
</div>

<div align="center">
    <center><b><i>Self-Directed Capstone Project for the MIT Applied Data Science Program</i></b></center>
</div>

A timeseries regression model was created to forecast future carbon emissions using timeseries data for electric energy production emissions from natural gas in the US from 1973 to 2016.

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
- A 3% mean absolute percent error (MAPE) for the final model was calculated against the validation dataset<br>

  ![](Images/Regression_Emissions/Carbon_Emissions_Validation.png)<br>
- Carbon emissions levels were forecasted for the 12 months following the provided data<br>

  ![](Images/Regression_Emissions/Carbon_Emissions_Forecast.png)

---
<!--                                *********************************** 3 - Clustering Cars ************************************************* -->

<div align="center">
    <center><h1>Automobile Clustering - Unsupervised Learning</h1></center>
</div>

<div align="center">
    <center><b><i>Weekly Project - MIT Applied Data Science Program</i></b></center>
</div>

A clustering model was built to group and identify similarities among automobile models sold between 1970 and 1982.  The provided vehicle data included a variety of characteristics including gas mileage, engine cylinders, horsepower and other vehicle specifications.<br>

### Major Accomplishments:
- Dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) were employed to group vehicles with similar characteristics
- Clusters identified with t-SNE were examined against the original variables to uncover common traits and correlations<br><br>
  ![](Images/Unsupervised_Cars/Box_plots.png)

### Key Outcomes:
- Clustering techniques identified that vehicles were largely grouped by engine cylinder count, displacement, horsepower, and vehicle weight which are all highly correlated<br>

  ![](Images/Unsupervised_Cars/PCA_coefficients.png)<br>
<br>
- Analysis also highlighted the effects of the gas crisis as a degradation in characteristics associated with engine power coincided with an increase in gas mileage<br>

  ![](Images/Unsupervised_Cars/PCA_Gradients_Cars.png)<br>
  ![](Images/Unsupervised_Cars/t-SNE_plots.png)

---
<!--                                *********************************** 4 - Boston Houses - Linear Regression ************************************************* -->

<div align="center">
    <center><h1>Boston Home Prices - Linear Regression Project</h1></center>
</div>

<div align="center">
    <center><b><i>Weekly Project - MIT Applied Data Science Program</i></b></center>
</div>

A linear regression model was created to predict median home values in Boston-area suburbs and towns based on a variety of factors including the geographic and socioeconomic characteristics of the home locations.  Housing data were provided and analysis was performed to determine the most important factors affecting values.  An equation was developed from the coefficients produced by the model capable of estimating home prices.
 
### Major Accomplishments:
- Performed exploratory data analysis of the provided data
- Examined variable correlations and removed multicollinearity using Variance Inflation Factor (VIF) criteria

  ![](Images/linear_regression_Boston_Homes/VIF_plots2.png)
  
- Performed tests on residuals to verify assumptions of linear regression

  ![](Images/linear_regression_Boston_Homes/Residual_Plots.png)

### Key Outcomes:
- This model achieved an R<sup>2</sup> value of .729 when cross validation techniques were applied
- Against the test dataset, the model had a Mean Absolute Percentage Error (MAPE) of 5.26%, very close to the 4.98% MAPE against the training dataset
- A linear regression equation was created from the coefficients produced by the models

<!--                                      *********************************** 5 - SVHN Neural Nets ************************************************* -->

---

<div align="center">
    <center><h1>Image Recognition - Neural Networks Project</h1></center>
</div>

<div align="center">
    <center><b><i>Weekly Project - MIT Applied Data Science Program</i></b></center>
</div>

Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) models were implemented to recognize numbers in a sample of 60,000 greyscale images derived from the Street View Housing Number (SVHN) dataset which contains images of numbers zero through nine.  Each neural network model was modified to improve prediction accuracy, and confusion matrices were created to assess the performance of the improved models.  A sample of the images is shown below.  The original SVHN source and citation are provided at the end of this project section.

![](Images/Neural_Networks_SVHN/SVHN_Samples.png)

### Major Accomplishments:
- ANN and CNN models were created using TensorFlow and Keras
- A simple ANN model was initially created using dense layers with ReLU activation.  An improved model incorporated higher node counts, additional dense layers, dropout and batch normalization layers, and a higher number of epochs
- A initial CNN model was created with convolutional, pooling, and dense layers using Leaky ReLU activation.  An improved model incorporated batch normalization, dropout, and additional convolution layers, while also increasing the number of epochs

### Key Outcomes:
- Artificial and Convolutional Neural Network models were created which achieved final accuracies of 75% and 90% respectively on the test data.  The confusion matrix for the CNN model is shown below:

  ![](Images/Neural_Networks_SVHN/Confusion_Matrix_SVHN.png)

***Original Source and Citation for SVHN Images Shown:***

Original Source:
[http://ufldl.stanford.edu/housenumbers](http://ufldl.stanford.edu/housenumbers)

Citation:  Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.


<!--                                      *********************************** Contact Info ************************************************* -->

---

<div align="center">
    <center><h1>Contact Information</h1></center>
</div>

- Phone: 978-417-1001
- Email:  drossetti12@gmail.com
- LinkedIn: [https://www.linkedin.com/in/daniel-r-10882139/](https://www.linkedin.com/in/daniel-r-10882139/)
- MIT ADSP Certificate:  [https://www.credential.net/0655323a-4a50-49d6-9dfc-a81e5e4b7ca8#gs.o5fack](https://www.credential.net/0655323a-4a50-49d6-9dfc-a81e5e4b7ca8#gs.o5fack)
