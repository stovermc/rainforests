from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.001, C=100.)

# train on all rows but last one
classifier.fit(digits.data[:-1], digits.target[:-1])

# predict on last row
classifier.predict(digits.data[-1:])

# compare to correct answer
digits.target[-1:]

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(digits.data[:-1], digits.target[:-1])

clf.predict(digits.data[-1:])
digits.target([-1:])

clf.predict(digits.data[0:1])
digits.target[0:1]
