# QUESTION :
'''Data of monthly experience and income distribution of different employs is given.
 Perform regression.'''
# Getting best result by Polynomial Regression
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv')
X = dataset.iloc[:, 0:1].values # months experience
y = dataset.iloc[:, 1].values   # income

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 7)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'm', label = 'Data points')
plt.plot(X,lin_reg.predict(X), color = 'b', label = 'Best fit')
plt.title('Months Experience vs Employee Income\n(Linear Regression)', fontweight = 'bold')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.legend()
plt.grid()
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'm', label = 'Data points')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'b', label = 'Best fit')
plt.title('Months Experience vs Employee Income\n(Polynomial Regression)',fontweight = 'bold')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.legend()
plt.grid()
plt.show()

# Predicting a new result with Polynomial Regression
y_pred1 = lin_reg_2.predict(poly_reg.fit_transform([[6]]))
print('\n\nPredicting Income with Polynomial Regression on 6 months of experience\nINCOME = ',y_pred1)

# Predicting a new result with Linear Regression
y_pred2 = lin_reg.predict([[6]])
print('\n\nPredicting Income with Polynomial Regression on 6 months of experience\nINCOME =  = ',y_pred2)