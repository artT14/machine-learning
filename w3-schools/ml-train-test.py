import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#PT1 create/supply data set
numpy.random.seed(2)
x = numpy.random.normal(3,1,100)
y = numpy.random.normal(150,40,100) / x

# plt.scatter(x,y)
# plt.show()

#PT2 split the data into training sets and testing sets
# TRAIN on 80%, TEST on 20%
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# plt.scatter(train_x, train_y)
# plt.show()

#PT3 fit the data set
mymodel = numpy.poly1d(numpy.polyfit(train_x,train_y, 4))

myline = numpy.linspace(0,6,100)

r2 = r2_score(train_y,mymodel(train_x))
print("r2:",r2)

r2_w_test_data = r2_score(test_y,mymodel(test_x))
print("r2 for test data:",r2_w_test_data)

# plt.scatter(train_x,train_y)
# plt.plot(myline,mymodel(myline))
# plt.show()

#PREDICT VALUES
print(mymodel(5))