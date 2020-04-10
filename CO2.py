# QUESTION :
'''Data of global production of CO2 of a place is given between 1970s to 2010. Predict the 
   CO2 production for the years 2011, 2012 and 2013 using the old data set.'''
# Getting best result by Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

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
plt.title('CO2 Production Using Old Data Set\n(Linear Regression)', fontweight = 'bold')
plt.xlabel('Year')
plt.ylabel('CO2 Produce')
plt.legend()
plt.grid()
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'm', label = 'Data points')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'b', label = 'Best fit')
plt.title('CO2 Production Using Old Data Set\n(Polynomial Regression)', fontweight = 'bold')
plt.xlabel('Year')
plt.ylabel('CO2 Produce')
plt.legend()
plt.grid()
plt.show()

# Predicting a new result with Linear Regression
y_pred1 = lin_reg.predict([[2011]])
y_pred2 = lin_reg.predict([[2012]])
y_pred3 = lin_reg.predict([[2013]])
print('\n\nCO2 Production in the given year by Linear Regression will be\n\t2011 = ',y_pred1,'\n\t2012 = ',y_pred2,'\n\t2013 = ',y_pred3)

# Predicting a new result with Polynomial Regression
y_pred4 = lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
y_pred5 = lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
y_pred6 = lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print('\n\nCO2 Production in the given year by Ploynomial Regression will be\n\t2011 = ',y_pred4,'\n\t2012 = ',y_pred5,'\n\t2013 = ',y_pred6)