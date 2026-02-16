# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets.
6.Create an SGD Regressor model.
7.Fit the model on the training data.
8.Evaluate the model performance.
9.Make predictions and visualize the results.


## Program:
```
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

data = data.drop(['car_ID' , 'CarName'], axis = 1)
data = pd.get_dummies(data, drop_first = True)

X = data.drop('price',axis = 1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size =0.2,random_state = 42)
sgd_model = SGDRegressor(max_iter = 1000, tol=1e-3)
sgd_model.fit(X_train ,y_train)
y_pred =sgd_model.predict(X_test)

mse  = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test , y_pred)

print("Name: gowtham u")
print("Reg. No: 25005013")
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-square':}:{r2_score(y_test,y_pred):}")

print("\nModel Coeffients: ")
print("Coefficient:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

plt.scatter(y_test , y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test) , max(y_test)], [min(y_test) , max(y_test)], color = 'red')
/*
Program to implement SGD Regressor for linear regression.
Developed by: gowtham u
RegisterNumber:  25005013
*/
```

## output:
<img width="922" height="297" alt="image" src="https://github.com/user-attachments/assets/a7768d8d-5cf5-4add-8c9e-2e108df78aad" />
<img width="1249" height="530" alt="Screenshot 2026-02-16 110849" src="https://github.com/user-attachments/assets/ca5316b0-a53f-4543-b14f-5fcb4f7737b7" />
<img width="579" height="651" alt="Screenshot 2026-02-16 110819" src="https://github.com/user-attachments/assets/b3949666-6fc7-43b7-9f03-d3677195c497" />
<img width="1332" height="389" alt="Screenshot 2026-02-16 110715" src="https://github.com/user-attachments/assets/2f4c176f-00cf-4427-ac34-37482b520ef9" />
<img width="1099" height="612" alt="Screenshot 2026-02-16 110903" src="https://github.com/user-attachments/assets/8caf0c77-e31c-4e73-82e1-ded6e0c99f95" />




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
