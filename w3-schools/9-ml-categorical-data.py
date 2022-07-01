"""
CATEGORICAL DATA:
When data has categories represented by strings, 
it may be necessary to transform the data to numerical so 
that model can learn
"""
import pandas as pd
from sklearn import linear_model

cars = pd.read_csv('cars.csv')
# print(cars.to_string())

# ONE HOT ENCODING
#encode strings as numbers instead
ohe_cars = pd.get_dummies(cars[['Car']])
# print(ohe_cars.to_string())

X = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
y = cars['CO2']

regr = linear_model.LinearRegression()
regr.fit(X,y)

##predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])

print(predictedCO2)

