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

    if show_probability:
        # Plot the probability gradient
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.colorbar(cs, label='Probability')

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the hyperplane

    plt.contour(xx, yy, Z, colors=['red', 'black', 'blue'], levels=[-1, 0, 1], alpha=1, linestyles=['-', '-', '-'])

    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="upper right", title="Classes")

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

    # Affichage des données
    title="Training data non linéairement séparable"
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.plot(X[:, 0], X[:, 1],'ko')
    #plt.savefig(f'plots/{title}.pdf')
    #plt.show()


    #classifier = SVC(kernel="rbf", probability=True)
    #classifier = classifier.fit(X, Y)

    #plot_data_hyperplan(X, Y, classifier, "Graph_SVM_non_lineaire_with_proba", show_probability=True,save=False)
    
    #plot_data_hyperplan(X, Y, classifier, "Graph_SVM_non_lineaire_without_proba", show_probability=False,save=False)

    # Définir le modèle et les hyperparamètres à tester
    model = SVC()

    # Partie 1 : Les nombres de 1 à 6
    part1 = list(range(1, 7))

    # Partie 2 : Les multiples de puissances de 10 (10, 20, ..., 90, 100, 200, ...)
    part2 = []
    for power in range(-2, 2):  # On peut ajuster la plage selon les besoins
        part2.append( 10 ** power)



    # Fusionner les deux parties
    result = part1 + part2 #+ list(np.logspace(-2,1))

    print(f"{result=}")

    param_grid = {
        'C': result,         # Paramètre de régularisation
        'degree': part1,        # Paramètre du noyau RBF
        'coef0' : part1 + part2,
        'kernel': ['poly'],               # Type de noyau
        'probability' : [True],
    }

    # Configuration de GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='precision', n_jobs=-1, verbose=2)

    # Recherche des meilleurs hyperparamètres
    grid_search.fit(X, Y)

    # Afficher les résultats
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure précision :", grid_search.best_score_)


    classifier = SVC(**grid_search.best_params_).fit(X,Y)

    print(f'{classifier.coef0=}')
    print(f'{classifier.class_weight=}')
    print(f'{classifier.dual_coef_=}')
    print(f'{classifier.support_vectors_=}')
    print(f'{classifier.get_params()=}')

    # Récupérer les paramètres du modèle
    support_vectors = classifier.support_vectors_.flatten()
    dual_coefs = classifier.dual_coef_[0]  # Alpha * y
    intercept = classifier.intercept_[0]
    gamma = classifier._gamma  # Valeur de gamma

    # Initialiser les coefficients
    a, b, c, d = 0, 0, 0, intercept

    # Calculer les contributions des vecteurs de support
    for coef, sv in zip(dual_coefs, support_vectors):
        # Expansion du noyau polynomial
        d += coef * (gamma * sv + 1) ** 3
        c += 3 * coef * (gamma * sv + 1) ** 2 * gamma
        b += 3 * coef * (gamma * sv + 1) * gamma**2
        a += coef * gamma**3

    # Résultat
    print(f"Les coefficients du polynôme sont : a={a}, b={b}, c={c}, d={d}")

    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_best_poly_non_lineaire_with_proba", show_probability=True,save=False)
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM-best_poly_non_lineaire_without_proba", show_probability=False,save=False)

if __name__ == "__main__":
    main()