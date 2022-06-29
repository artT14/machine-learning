from matplotlib.pyplot import sca
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

"""
MULTIPLE REGRESSION:
predicting value based on 2 or more variables
"""
# #cars.csv contains various info about various cars

# df = pandas.read_csv("cars.csv")
# X = df[['Weight', 'Volume']] #INDEPENDENT vars
# y = df['CO2'] #DEPENDENT vars

# regr = linear_model.LinearRegression()
# regr.fit(X, y)

# #predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
# predictedCO2 = regr.predict([[2300, 1300]])

# print(predictedCO2) #SHOULD OUTPUT [107.2087328]
#  #print coefficient
# print(regr.coef_) # SHOULD OUTPUT [0.00755095 0.00780526]
# # means that if weight increases by 1kg, CO2 increase by 0.00755095
# #  "      "  if volume increases by 1cm^3, CO2 increase by 0.00780526

"""
SCALE:
useful for scaling down values using the following formula
z is the scaled value, x is the original value, u is the mean, s is the standard deviation
z = (x - u) / s
can do this automatically using StandardScaler() in sklearn module
"""

# scale = StandardScaler() # initialize the scaler

# df = pandas.read_csv("cars2.csv") # read CSV table

# X = df[['Weight', 'Volume']] # fetch Weight & Volume columns to X

# scaledX = scale.fit_transform(X) # scale down X using the scaler

# print(scaledX)

"""
PREDICTING CO2 using scaled values
"""

scale = StandardScaler()

df = pandas.read_csv("cars2.csv")

X = df[['Weight','Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX,y)

scaled = scale.transform([[2300,1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)