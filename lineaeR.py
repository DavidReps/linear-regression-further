import numpy
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#  Use the linear_model.LinearRegression class to build predictive models based
# on the first N data examples, for N = 10d, d = 1, 2, · · · , 50. Plot the goodnessof-fit of these models as functions of the data sample size, N, evaluated in two
# different ways: based on the full training sample of size N (for each N), and
# using 10-fold cross-validation. The plot will, therefore, show two curves. The
# cross-validation curve provides an approximation of generalization performance.
# (b) Describe your results from the preceding part. As functions of the sample size,
# how are the behaviors of the in-sample and cross-validation curves different qualitatively at smaller sample sizes? Are these behaviors as expected based on our
# discussions from class? Explain. What happens at larger sample sizes in the given
# range? At approximately what sample size does the difference between the two
# performance curves first appear to be starting to stabilize? Include additional
# plots if you feel that they help in establishing your points.

#code for # QUESTION:  3
D = numpy.loadtxt('regressionDataPS3.txt', delimiter = ',')

prediction = []
CrossVal = []

y = D[:,-1]
X = D[:,:-1]

for n in range(10, 510, 10):

    rowsx = X[0:n,:]
    rowsy = y[0:n]

    model = LinearRegression()
    model.fit(rowsx,rowsy)
    temp = model.score(rowsx, rowsy)

    prediction.append(temp)

    CrossVal.append(numpy.mean(cross_val_score(model, rowsx, rowsy, cv=10)))

plt.title("Breast Cancer precition")
plt.ylabel("Accuracy")
plt.xlabel("Iteration number")
plt.plot(CrossVal)
plt.plot(prediction)
plt.show()
#end Q 3
