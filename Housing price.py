# QUESTION :
'''Housing price according to the ID is assigned to every-house. Perform future analysis
 where when ID is inserted the housing price is displayed.'''
# Getting best result by Linear Regression
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.figure(figsize=(8,4))
plt.scatter(X, y, color = 'm', label = 'Data points')
plt.plot(X,lin_reg.predict(X), color = 'b', label = 'Best fit')
plt.title('Predicting House Price Using ID\n(Linear Regression)', fontweight = 'bold')
plt.xlabel('ID')
plt.ylabel('Sale Price')
plt.legend()
plt.show()

# Visualising the Polynomial Regression results
plt.figure(figsize=(8,4))
plt.scatter(X, y, color = 'm', label = 'Data points')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'b', label = 'Best fit')
plt.title('Predicting House Price Using ID\n(Polynomial Regression)', fontweight = 'bold')
plt.xlabel('ID')
plt.ylabel('Sale Price')
plt.legend()
plt.show()

# Predicting a new result with Linear Regression
y_pred1 = lin_reg.predict([[1500]])
print('\nPredicting House price with Linear Regression when ID is 1500\nHOUSE PRICE =  ',y_pred1)

# Predicting a new result with Polynomial Regression
y_pred2 = lin_reg_2.predict(poly_reg.fit_transform([[1500]]))
print('\n\nPredicting House price with Polynomial Regression when ID is 1500\nHOUSE PRICE = ',y_pred2)
