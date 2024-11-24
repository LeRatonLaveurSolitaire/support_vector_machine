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
    Y = np.concatenate((np.ones((n1, 1)), -1 * np.ones((n2, 1))))[:, 0]
    return X, Y


def plot_data_hyperplan(X, Y, classifier, title, show_probability=False, save=False):
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Plot the decision boundary
    if show_probability:
        # Plot the probability gradient
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.colorbar(label='Probability')
    # else:
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    #plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                colors=['#FFAAAA', '#AAAAFF', '#AAFFAA'])

    # Plot the hyperplane
    plt.contour(xx, yy, Z, colors=['red', 'black', 'blue'], levels=[-1, 0, 1], alpha=1,
                linestyles=['-', '-', '-'])

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Plot the support vectors
    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', alpha=0.5)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if save:
        plt.savefig(f'plots/{title}.pdf')
    plt.show()
    plt.close()

def main():
    X, Y = genere_ex_1()
    classifier = SVC(kernel="linear", probability=True)
    classifier = classifier.fit(X, Y)

    w = classifier.coef_[0]
    b = classifier.intercept_[0]

    print(f"{w=}")
    print(f"{b=}")

    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_lineaire_with_proba", show_probability=True,save=False)
    
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_lineaire_without_proba", show_probability=False,save=False)


if __name__ == "__main__":
    main()
