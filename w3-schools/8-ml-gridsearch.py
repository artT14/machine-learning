"""
GRID SEARCH:
try out different values for parameters and then pick the value that gives the best score.
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# load some default dataset from sklearn
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

logit = LogisticRegression(max_iter = 10000)
print(logit.fit(X,y))
# Default C of 1.0
print(logit.score(X,y))

# C is a paramater in LogisticRegression
C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
scores = []

for choice in C:
	logit.set_params(C=choice)
	logit.fit(X, y)
	scores.append(logit.score(X, y))

print(scores)
