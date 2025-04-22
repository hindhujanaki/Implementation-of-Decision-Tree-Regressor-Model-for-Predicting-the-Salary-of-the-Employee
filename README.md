# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: G.HINDHU
RegisterNumber: 212223230079
*/
```
```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```
## Output:
### Data Head:
![image](https://github.com/user-attachments/assets/aabef409-74e0-42f1-bb8c-71e58076933e)

### Data Info:
![image](https://github.com/user-attachments/assets/cbc69aad-dba0-47bf-9c6f-c2f1840f0e81)

### isnull() sum():
![image](https://github.com/user-attachments/assets/967744d5-89bd-4409-9d9c-76497ddafdc5)

### Data Head for salary:
![image](https://github.com/user-attachments/assets/504bc3ab-c601-41bd-a696-02a23d9134e4)


### Mean Squared Error :
![image](https://github.com/user-attachments/assets/2fad7591-276c-45b5-9f5d-c2abbdb5e456)


### r2 Value:
![image](https://github.com/user-attachments/assets/21d58cab-e962-4712-94a8-6601a826ffaa)


### Data prediction :

![image](https://github.com/user-attachments/assets/fd64edab-80d0-46e0-a974-9a9abe68c90e)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
