import numpy
from sklearn import metrics
import matplotlib.pyplot as plt

# What is a confusion matrix?
"""
It is a table that is used in classification problems to assess where errors in the model were made.

The rows represent the actual classes the outcomes should have been. While the columns represent the
predictions we have made. Using this table it is easy to see which predictions are wrong.
"""

#pseudo-actual values
actual = numpy.random.binomial(1, 0.9, size = 1000)

#pseudo-predicted values
predicted = numpy.random.binomial(1, 0.9, size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

"""
ACCURRACY
(True Positive + True Negative) / Total Predictions
"""

Accuracy = metrics.accuracy_score(actual, predicted)


"""
PRECISION
True Positive / (True Positive + False Positive)
"""

Precision = metrics.precision_score(actual, predicted)



"""
SENSITIVITY (RECALL)
True Positive / (True Positive + False Negative)
"""

Sensitivity_recall = metrics.recall_score(actual, predicted)



"""
SPECIFICITY
True Negative / (True Negative + False Positive)
"""

Specificity = metrics.recall_score(actual, predicted, pos_label=0)


"""
F-SCORE
the "harmonic mean" of precision and sensitivity.
2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
"""

F1_score = metrics.f1_score(actual, predicted)

print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})