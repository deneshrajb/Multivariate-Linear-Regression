# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br> Import required Python libraries (numpy, matplotlib, sklearn) and load the California Housing dataset using fetch_california_housing().


### Step2
<br> Extract the features (X) and target values (y) from the dataset, then split them into training and testing sets using train_test_split() (e.g., 60% train, 40% test).


### Step3
<br> Create a LinearRegression() model and fit it using the training data (X_train, y_train) to learn the relationship between input features and target output.


### Step4
<br> Use the trained model to predict target values for the test set (X_test) and evaluate its performance using the R² score (reg.score()), which measures how well the model explains the variability of the data.


### Step5
<br> Print the coefficients, intercept, and variance score. Optionally, visualize the actual vs. predicted values using a scatter plot to assess the model’s accuracy.


## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
boston = datasets.fetch_california_housing()

X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
print("Variance score (R^2): {:.2f}".format(reg.score(X_test, y_test)))

```
## Output:
Coefficients: [ 4.36676927e-01  9.58588561e-03 -9.61859974e-02  5.77800722e-01
 -4.33084487e-06 -3.13805195e-03 -4.22347516e-01 -4.34869554e-01]
Intercept: -36.93649134035418
Variance score (R^2): 0.61

### Insert your output

<img width="656" height="76" alt="{F66D175B-0E91-4873-BE68-FBF99C2429AB}" src="https://github.com/user-attachments/assets/16806061-026a-4f7e-88fe-b77fe0d8b6ac" />
<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
