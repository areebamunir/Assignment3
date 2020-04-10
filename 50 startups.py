#QUESTION :
'''Take 50 startups of any two countries and find out which country is going to provide best
 profit in future.'''
# Using Decision Tree Regression for the best results 
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Customerize rows and column for New York
RS = dataset.loc[(dataset['State'] == 'New York'), ['R&D Spend']]
AS = dataset.loc[(dataset['State'] == 'New York'), ['Administration']]
MS = dataset.loc[(dataset['State'] == 'New York'), ['Marketing Spend']]

# Add all three spend features into one feature
dataset['Total Spend']= dataset['R&D Spend'] + dataset['Administration'] + dataset['Marketing Spend'  ]         # x-axis
TS = dataset.loc[(dataset['State'] == 'New York'), ['Total Spend']] # x-axis
P = dataset.loc[(dataset['State'] == 'New York'), ['Profit']]       # y-axis

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(TS, P)

# Visualising the Decision Tree Regression results 
plt.scatter(TS, P, color = 'm',label = 'Data ponits')
plt.plot(TS, regressor.predict(TS), color = 'b', label = 'Best fit')
plt.title('Total Spend vs Profit (For New York)\nDecision Tree Regression', fontweight = 'bold')
plt.xlabel('Spend')
plt.ylabel('Profit')
plt.legend()
plt.grid()
plt.show()

# Customerize rows and column for California
Rs = dataset.loc[(dataset['State'] == 'California'), ['R&D Spend']]
As = dataset.loc[(dataset['State'] == 'California'), ['Administration']]
Ms = dataset.loc[(dataset['State'] == 'California'), ['Marketing Spend']]

# Add all three spend features into one feature
dataset['Total Spend']= dataset['R&D Spend'] + dataset['Administration'] + dataset['Marketing Spend'  ]         # x-axis
Ts = dataset.loc[(dataset['State'] == 'California'), ['Total Spend']] # x-axis
p = dataset.loc[(dataset['State'] == 'California'), ['Profit']]       # y-axis

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Ts, p)

# Visualising the Decision Tree Regression results 
plt.scatter(Ts, p, color = 'm',label = 'Data ponits')
plt.plot(Ts, regressor.predict(Ts), color = 'b', label = 'Best fit')
plt.title('Total Spend vs Profit (For California)\nDecision Tree Regression', fontweight = 'bold')
plt.xlabel('Spend')
plt.ylabel('Profit')
plt.legend()
plt.grid()
plt.show()
