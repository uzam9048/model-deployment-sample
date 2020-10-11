# Simple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import pickle


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

#dump into Pickle
pickle.dump(regressor,open('model.pkl','wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9]]))

