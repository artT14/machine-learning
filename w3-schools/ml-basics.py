#FOLLOWING python ML tutorials on w3schools.com
import numpy
import math
from scipy import stats
import matplotlib.pyplot as plt

"""
DATA SETS:
-----------
Machine Learning Deals w/ DATA SETS
"""

"""
DATA TYPES:
-----------
> NUMERICAL
DISCRETE (integers)
CONTINUOUS (flaots,doubles)

> CATEGORICAL
(values that cannot be measured against each other)
(color values, yes/no answers)

> ORDINAL
(like categorical but can be measured against each other)
(grades A > B)
"""

"""MEAN/MEDIAN/MODE"""
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

"""
MEAN:
(99+86+87+88+111+86+103+87+94+78+77+85+86) / 13 = 89.77
"""

mean = numpy.mean(speed)

print("SPEED:",speed)
print("MEAN:",mean)

"""
MEDIAN:
77, 78, 85, 86, 86, 86, 87, 87, 88, 94, 99, 103, 111
                        ^^
"""

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

median = numpy.median(speed)

print("SPEED:",speed)
print("MEDIAN:",median)

"""
MODE:
99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 = 86
    ^^               ^^                           ^^
"""
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

mode = stats.mode(speed)

print("SPEED:",speed)
print("MODE:",mode)

"""
STD DEV:
(how spread out the values are)
"""

speed = [86,87,88,86,87,85,86]

std_dev = numpy.std(speed)

print("SPEED:",speed)
print("STD DEV:",std_dev)

speed = [32,111,138,28,59,77,97]

std_dev = numpy.std(speed)

print("SPEED:",speed)
print("STD DEV:",std_dev)

"""
VARIANCE:
(another value that indicates how spread out the values are)
(look up definition if you need it)
"""

speed = [32,111,138,28,59,77,97]

variance = numpy.var(speed)

print("SPEED:",speed)
print("VARIANCE:",variance)
print("STD_DEV:",math.sqrt(variance))


"""
std_dev = sqrt(variance)
"""

"""
PERCENTILE:

"""

ages =[5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

percentile = numpy.percentile(ages, 75)

print("SPEED:",speed)
print("PERCENTILE:",percentile)
# 75% of people are 43 y/o or younger

"""
CREATE RANDOM DATA SET
"""

rand_set = numpy.random.uniform(0.0, 5.0, 250)

#print("RANDOM SET:",rand_set)

"""
DISPLAY HISTOGRAM
"""
# x = numpy.random.uniform(0.0, 5.0, 250)
# plt.hist(x, 5)
# plt.show()

"""
NORMAL DISTRIBUTION
"""
# x = numpy.random.normal(5.0, 1.0, 100000)
# plt.hist(x, 100)
# plt.show()

"""
SCATTER PLOT
"""
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# plt.scatter(x, y)
# plt.show()

""""""
# x = numpy.random.normal(5.0, 1.0, 1000)
# y = numpy.random.normal(10.0, 2.0, 1000)
# plt.scatter(x, y)
# plt.show()

"""
LINEAR REGERSSION:
useful for predicting future values
"""
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

print(r) #r indicates how accurate the prediction is

speed = myfunc(10) #predicting the speed of a 10 y/o car
print(speed)
