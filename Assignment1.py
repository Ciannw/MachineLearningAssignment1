import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# converts the data into a numpy array.
# Adds appropriate labels to the data.
# Then Merges testing and training each into their own array
def main():
    names = ['X', "Y"]
    test_sDat = pd.read_csv('test.sDAT.csv', names=names)
    train_sDat = pd.read_csv('train.sDAT.csv', names=names)
    test_sNC = pd.read_csv('test.sNC.csv', names=names)
    train_sNC = pd.read_csv('train.sNC.csv', names=names)
    grid = pd.read_csv('2D_grid_points.csv').to_numpy()
    test_sDat = test_sDat.assign(label=1)
    train_sDat = train_sDat.assign(label=1)
    test_sNC = test_sNC.assign(label=0)
    train_sNC = train_sNC.assign(label=0)
    test_set = pd.concat([test_sDat, test_sNC], axis=0)
    train_set = pd.concat([train_sDat, train_sNC], axis=0)
    return modeler(test_set, train_set, grid),modeler(test_set, train_set, grid,metric="manhattan",k=30)


def modeler(test_set, train_set, grid,metric='euclidean', k=0):
    tr_data = train_set[['X', "Y"]]
    te_data = test_set[['X', "Y"]]
    tr_label = train_set[['label']]
    te_label = test_set[['label']]
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    if k != 0:
        k_values = [k]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(tr_data, tr_label)
        knn.predict(te_data)
        grid_predicted_labels = knn.predict(grid)
        grid_predicted_labels = np.reshape(grid_predicted_labels, (grid_predicted_labels.shape[0], 1))
        grid2 = np.concatenate([grid, grid_predicted_labels], axis=1)
        colors = ["green" if label == 0 else "blue" for label in train_set["label"]]
        colors2 = ["green" if label == 0 else "blue" for label in test_set["label"]]
        colors3 = ["green" if label == 0 else "blue" for label in grid2[:, 2]]
        plt.scatter(grid2[:, 0], grid2[:, 1], c=colors3, marker='.')
        plt.scatter(train_set["X"], train_set["Y"], c=colors, marker='o')
        plt.scatter(test_set["X"], test_set["Y"], c=colors2, marker='+')
        if len(k_values) > 1:
            plt.title("KNN with k = {} Train Error ={} Test Error ={}".format(k, round(1 - knn.score(tr_data, tr_label), 5),round(1 - knn.score(te_data, te_label), 5)))
        else:
            plt.title("Manhattan KNN with k = {} Train Error ={} Test Error ={}".format(k, round(1 - knn.score(tr_data, tr_label), 5),round(1 - knn.score(te_data, te_label), 5)))

        if len(k_values) == 1:
            plt.savefig('{}.jpg'.format("manhattan"), dpi=300)
        else:
            plt.savefig('{}.jpg'.format(k), dpi=300)


main()
