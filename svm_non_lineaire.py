import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def genere_ex_2(n=300, mu = [0,0], std=0.25, delta = 0.2):
    X = np.random.multivariate_normal(mu, np.diagflat(std*np.ones(2)),n)
    Y = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        x = X[i,0]
        y = X[i,1]
        if y < x*(x-1)*(x+1):
            Y[i] = -1
            X[i,1] = X[i,1] - delta
        else:
            Y[i] = 1
            X[i,1] = X[i,1] + delta
    
    return X,Y

def plot_data_hyperplan(X, Y, classifier, title, show_probability=False,save=False):
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    plt.figure(figsize=(10, 8))

    # Determine if the kernel is linear
    is_linear = classifier.kernel == 'linear'

    if show_probability:
        # Plot the probability gradient
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.colorbar(cs, label='Probability')

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                          colors=['#FFAAAA', '#AAAAFF', '#AAFFAA'])

    # Plot the hyperplane
    if is_linear:
        w = classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(x_min, x_max)
        yy = a * xx - (classifier.intercept_[0]) / w[1]
        plt.plot(xx, yy, 'k-')
    else:
        plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Plot the support vectors
    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', alpha=0.5)


    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="upper right", title="Classes")
    #plt.add_artist(legend1)

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
    X, Y = genere_ex_2()
    classifier = SVC(kernel="rbf", probability=True)
    classifier = classifier.fit(X, Y)

    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_non_lineaire_with_proba", show_probability=True,save=False)
    
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_non_lineaire_without_proba", show_probability=False,save=False)

    # Définir le modèle et les hyperparamètres à tester
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],         # Paramètre de régularisation
        'gamma': [1, 0.1, 0.01, 0.001], # Paramètre du noyau RBF
        'kernel': ['rbf']               # Type de noyau
    }

    # Configuration de GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    # Recherche des meilleurs hyperparamètres
    grid_search.fit(X, Y)

    # Afficher les résultats
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure précision :", grid_search.best_score_)

if __name__ == "__main__":
    main()