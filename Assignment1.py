import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# converts the data into a numpy array.
# Adds appropriate labels to the data.
# Then Merges testing and training each into their own array
def data_loader():
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
    return runner(test_set, train_set, grid)


def runner(test_set, train_set, grid):
    tr_data = train_set[['X', "Y"]].to_numpy()
    te_data = test_set[['X', "Y"]].to_numpy()
    tr_label = train_set[['label']].to_numpy()
    te_label = test_set[['label']].to_numpy()
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(tr_data, tr_label)
        knn.predict(te_data)
        grid_predicted_labels = knn.predict(grid)
        print(grid.shape)
        print(grid_predicted_labels.shape)
        grid_predicted_labels = np.reshape(grid_predicted_labels, (grid_predicted_labels.shape[0], 1))
        grid2 = np.concatenate([grid, grid_predicted_labels], axis=1)
        print()
        colors = ["green" if label == 0 else "blue" for label in train_set["label"]]
        colors2 = ["green" if label == 0 else "blue" for label in test_set["label"]]
        colors3 = ["green" if label == 0 else "blue" for label in grid2[:, 2]]
        plt.scatter(grid2[:, 0], grid2[:, 1], c=colors3, marker='.')
        plt.scatter(train_set["X"], train_set["Y"], c=colors, marker='o')
        plt.scatter(test_set["X"], test_set["Y"], c=colors2, marker='+')
        plt.title("KNN with k = {} Train Error ={} Test Error ={}".format(k, round(1 - knn.score(tr_data, tr_label), 3),
                                                                          round(1 - knn.score(te_data, te_label), 3)))
        plt.show()




data_loader()
