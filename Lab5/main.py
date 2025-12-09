import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import stats

import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv('data.csv')
features_to_check = ['year', 'mileage']
z = np.abs(stats.zscore(df[features_to_check]))
df = df[(z < 5).all(axis=1)].copy()


class KMeansMy:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.tol = 1e-4
        self.centre = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centre = X[idx]

        for i in range(100):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centre, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centre = np.array([
                X[self.labels == k].mean(axis=0) if np.sum(self.labels == k) > 0 else X[np.random.choice(X.shape[0])]
                for k in range(self.n_clusters)
            ])

            if np.all(np.abs(new_centre - self.centre) < self.tol):
                self.centre = new_centre
                break

            self.centre = new_centre

        dists = np.linalg.norm(X[:, np.newaxis] - self.centre, axis=2)
        min_dists = np.min(dists, axis=1)
        self.inertia_ = np.sum(min_dists ** 2)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centre, axis=2)
        return np.argmin(distances, axis=1)


le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])

features = ['price', 'mileage', 'year']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
k_range = range(1, 10)
for k in k_range:
    km = KMeansMy(n_clusters=k)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('wcss')
plt.show()

custom_km = KMeansMy()
custom_km.fit(X_scaled)
custom_labels = custom_km.labels

sklearn_km = KMeans(n_clusters=3, random_state=42, n_init=10)
sklearn_labels = sklearn_km.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=custom_labels, cmap='viridis')
axes[0].set_title('Власний KMeans')

scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=sklearn_labels, cmap='viridis')
axes[1].set_title('Sklearn KMeans')

scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='viridis')
axes[2].set_title('Sklearn Agglomerative Clustering')

plt.tight_layout()
plt.show()
