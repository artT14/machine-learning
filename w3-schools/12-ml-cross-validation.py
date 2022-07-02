"""
CROSS VALIDATION:
When adjusting models we are aiming to increase overall model performance on unseen data.
Hyperparameter tuning can lead to much better performance on test sets. However, optimizing
parameters to the test set can lead information leakage causing the model to preform worse
on unseen data. To correct for this we can perform cross validation.

To better understand CV, we will be performing different methods on the iris dataset.
Let us first load in and separate the data.
"""
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, cross_val_score

X, y = datasets.load_iris(return_X_y=True)


"""
K-FOLD CROSS VALIDATION:
The training data used in the model is split, into k number of smaller sets, to be used to validate the model.
The model is then trained on k-1 folds of training set. The remaining fold is then used as a validation set to evaluate the model.
"""

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

"""
STRATIFIED K-FOLD:
In cases where classes are imbalanced we need a way to account for the imbalance
in both the train and validation sets. To do so we can stratify the target classes,
meaning that both sets will have an equal proportion of all classes.
"""
clf = DecisionTreeClassifier(random_state=42)

sk_folds = StratifiedKFold(n_splits = 5)

scores = cross_val_score(clf, X, y, cv = sk_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

"""
LEAVE-ONE-OUT(LOO):
nstead of selecting the number of splits in the training data set like k-fold LeaveOneOut,
utilize 1 observation to validate and n-1 observations to train. This method is an exaustive technique.
"""
clf = DecisionTreeClassifier(random_state=42)

loo = LeaveOneOut()

scores = cross_val_score(clf, X, y, cv = loo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


"""
LEAVE-P-OUT:
Leave-P-Out is simply a nuanced diffence to the Leave-One-Out idea,
in that we can select the number of p to use in our validation set.
"""

clf = DecisionTreeClassifier(random_state=42)

lpo = LeavePOut(p=2)

scores = cross_val_score(clf, X, y, cv = lpo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

"""
SHUFFLE SPLIT:
Unlike KFold, ShuffleSplit leaves out a percentage of the data,
not to be used in the train or validation sets. To do so we must
decide what the train and test sizes are, as well as the number of splits.
"""

clf = DecisionTreeClassifier(random_state=42)

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)

scores = cross_val_score(clf, X, y, cv = ss)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
