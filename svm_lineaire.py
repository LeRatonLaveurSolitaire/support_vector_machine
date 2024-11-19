import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def genere_ex_1(n1=100, n2=50, mu1=[0, 3], mu2=[3, 0], sd1=0.15, sd2=0.2):
    X = np.concatenate(
        (
            np.random.multivariate_normal(mu1, np.diagflat(sd1 * np.ones(2)), n1),
            (np.random.multivariate_normal(mu2, np.diagflat(sd2 * np.ones(2)), n2)),
        )
    )
    Y = np.concatenate((np.ones(n1, 1)), -1 * np.ones((n2, 1)))[:, 0]
    return X, Y


def main():
    X, Y = genere_ex_1()
    classifier = SVC(kernel="linear", probability=True)
    classifier = classifier.fit(X, Y)

    w = classifier.coef_[0]
    b = classifier.intercept_[0]

    # plot_data_hyperplan(X, Y, classifier, "Graph_SVM_lineaire")


if __name__ == "__main__":
    main()
