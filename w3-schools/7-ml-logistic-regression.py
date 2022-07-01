"""
LOGISTIC REGRESSION:
> aims to solve classification problems
> predicts categorical outcomes
(linear regression predicts continuous outcomes)

BINOMIAL: two outcomes (e.g. cat/dog)

MULTINOMIAL: more than 2 outcomes (3 species of an iris flower) 
"""
import numpy
from sklearn import linear_model

#X represents the size of a tumor in centimeters.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
# i.e. Transpose of X
#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:	
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))

print(predicted)

"""
PROBABILITY:
The coefficient and intercept values can be used to find the probability that each tumor is cancerous.
"""

def logit2prob(logr,x):
	log_odds = logr.coef_ * x + logr.intercept_
	odds = numpy.exp(log_odds)
	probability = odds / (1 + odds)
	return(probability)

print(logit2prob(logr, X))
