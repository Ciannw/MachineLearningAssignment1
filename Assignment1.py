import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def euclidian_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)


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
    for col in train_set.columns:
        print(col)
    return runner(test_set, train_set,grid)


def runner(test_set, train_set,grid):
    tr_data = train_set[['X', "Y"]].to_numpy()
    te_data = test_set[['X', "Y"]].to_numpy()
    tr_label = train_set[['label']].to_numpy()
    te_label = test_set[['label']].to_numpy()
    k_values  = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200 ]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(tr_data, tr_label)
        knn.predict(te_data)
        print(k)
        print(knn.score(te_data, te_label))
    colors = ["green" if label == 0 else "blue" for label in train_set["label"]]
    plt.scatter(train_set["X"],train_set["Y"],c = colors)
    plt.show()
data_loader()
