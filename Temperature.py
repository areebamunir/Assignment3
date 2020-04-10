# QUESTION :
'''Annual temperature between two industries is given. Predict the temperature in 2016 
and 2017 using the past data of both country.'''
# Getting best result by Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset for GCAG
dataset = pd.read_csv('annual_temp.csv')
X = dataset.loc[(dataset['Source'] == 'GCAG'), ['Year']]
y = dataset.loc[(dataset['Source'] == 'GCAG'), ['Mean']]

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the GCAG results
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (8,4))
fig.suptitle('Temperature vs Year (GCAG)\n', fontweight = 'bold')
ax1.set_title('Linear Regression')
ax1.scatter(X, y, color = 'm',)
ax1.plot(X, lin_reg.predict(X), color = 'b')
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Temperature')
ax2.set_title('Polynomial Regression')
ax2.scatter(X, y, color = 'm',)
ax2.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'b')
ax2.set_xlabel('Year')

for ax in fig.get_axes():
    ax.label_outer()

# Importing the dataset for GISTEMP
dataset = pd.read_csv('annual_temp.csv')
A = dataset.loc[(dataset['Source'] == 'GISTEMP'), ['Year']]
B = dataset.loc[(dataset['Source'] == 'GISTEMP'), ['Mean']]

# Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
A_poly = poly_reg.fit_transform(A)
poly_reg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the GISTEMP results
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (8,4))
fig.suptitle('Temperature vs Year (GISTEMP)\n', fontweight = 'bold')
ax1.set_title('Linear Regression')
ax1.scatter(A, B, color = 'm',)
ax1.plot(A, lin_reg.predict(A), color = 'b')
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Temperature')
ax2.set_title('Polynomial Regression')
ax2.scatter(A, B, color = 'm',)
ax2.plot(X,lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'b')
ax2.set_xlabel('Year')

for ax in fig.get_axes():
    ax.label_outer()

# Predicting a new result for GCAG
y_pred1 = lin_reg.predict([[2016]])
y_pred2 = lin_reg.predict([[2017]])
y_pred3 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
y_pred4 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('Predicting Temperature for GCAG for the years given below\nBY LINEAR REGRESSION : \n\t2016 = ',y_pred1,'\n\t2017 = ',y_pred2)
print('\nBY POLYNOMIAL REGRESSION : \n\t2016 = ',y_pred3,'\n\t2017 = ',y_pred4)

# Predicting a new result for GISTEMP
B_pred1 = lin_reg.predict([[2016]])
B_pred2 = lin_reg.predict([[2016]])
B_pred3 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
B_pred4 = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('\n\nPredicting Temperature for GISTEMP for the years given below\nBY LINEAR REGRESSION :  \n\t2016 = ',B_pred1,'\n\t2017 = ',B_pred2)
print('\nBY POLYNOMIAL REGRESSION : \n\t2016 = ',B_pred3,'\n\t2017 = ',B_pred4)
