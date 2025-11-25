import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from scipy import stats
import matplotlib


matplotlib.use('TkAgg')


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x',)
    colors = ('red', 'blue',)
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')


df = pd.read_csv('data.csv')
features_to_check = ['year', 'mileage']
z = np.abs(stats.zscore(df[features_to_check]))
df_clean = df[(z < 5).all(axis=1)].copy()


median_price = df_clean['price'].median()
df_clean['High_Price_Class'] = (df_clean['price'] > median_price).astype(int)

X_cars = df_clean[['year', 'mileage']].values
y_cars = df_clean['High_Price_Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_cars, y_cars, test_size=0.3, random_state=42, stratify=y_cars)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='rbf', random_state=42, gamma=1.0, C=1.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(10, 6))
plot_decision_regions(X_combined_std, y_combined, classifier=svm)
plt.xlabel('Рік випуску')
plt.ylabel('Пробіг')
plt.legend(labels=['Низька ціна', 'Висока ціна'])
plt.show()
