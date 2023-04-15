import mglearn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore")

# Классификатор BernoulliNB - считает не нулевые частоты признаков по каждому классу
X = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
Y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(Y):
    counts[label] = X[Y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))

clf = BernoulliNB()
clf.fit(X, Y)
print("clf.predict:\n{}".format(clf.predict(X[2:3])))

# Классификатор MultinomialNB-модель обучается на случайно сгенерированных данных и предсказывает класс для конкретного примера.
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
clf = MultinomialNB()
clf.fit(X, y)
print(clf.predict(X[2:3]))

# Классификатор GaussianNB - В этом примере модель обучается на наборе точек в двумерном пространстве и предсказывает класс для новой точки.
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))
