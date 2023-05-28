# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mohanapriya U
RegisterNumber: 212220040091
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
#splitting train and test data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying the predicted values
y_pred
y_test
#graph plot for traing data
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color='orange')
plt.title("Hours vs Scores(training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)


## Output:
![simple linear regression model for predicting the marks scored](sam.png)

1.df.head()

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/aea87724-4143-46ba-ba52-72fd341a7730)

2.df.tail()

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/58e5ed31-1eb4-4ae0-8778-0471dc7336bf)

3.Array value of X

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/c6cea979-c6bb-4b41-b898-b58b8b2e77a3)

4.Array value of Y

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/e0631c82-2ad0-4802-8f23-7d4ad5289ae1)

5.Values of Y prediction

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/578efe21-f384-4f96-bfbb-94e3808889a5)

6.Array valued of Y set

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/d4986b5e-fb18-40d6-a8c0-45598ee4d7df)

7.Training Set Graph

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/308c95b5-0e8a-45b9-ab4e-62a201d63902)

8.Values of MSE,MAE and RMSE

![image](https://github.com/MohanapriyaU76/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133958624/ec1445e4-4bdd-4311-a5ca-49918916a8f0)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
