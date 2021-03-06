"""
DECISION TREES:
A Decision Tree is a Flow Chart, and can help you make decisions based on previous experience.

In the example, a person will try to decide if he/she should go to a comedy show or not.

Luckily our example person has registered every time there was a comedy show in town,
and registered some information about the comedian, and also registered if he/she went or not.
"""
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("shows.csv")

#convert non-numerical data to numerical using LUTs
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

print(df)

# features are the columns we predict from
# targets are the columns we try to predict

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features] # feature column
y = df['Go'] # target column


print(X)
print(y)

# Creating a decision tree

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

# Gini = 1 - (x/n)^2 - (y/n)^2

# PREDICT values
# Should I go see a show starring a 40 years old American comedian,
#  with 10 years of experience, and a comedy ranking of 7?
print(dtree.predict([[40, 10, 7, 1]])) # YES
print(dtree.predict([[40, 10, 6, 1]])) # NO

# NOTE: decision tree will give different answers if ran enough times

