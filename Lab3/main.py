import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats


df = pd.read_csv('data.csv')[['year', 'mileage']].dropna()
z = np.abs(stats.zscore(df))
df = df[(z < 5).all(axis=1)]


df['label'] = pd.qcut(df['mileage'], q=3, labels=False, duplicates='drop')
X = df[['year', 'mileage']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=1)


class DecisionTree:
    def __init__(self):
        self.max_depth = 5
        self.tree_ = None

    def gini(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def entropy(self, y):
        classes = np.unique(y)
        ent = 0
        for c in classes:
            p = np.sum(y == c) / len(y)
            ent -= p * np.log2(p)
        return ent

    def best_split(self, X, y):
        best_feat, best_thresh, best_gain = None, None, -1
        current_impurity = self.gini(y)

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for t in values:
                left_idx = X[:, feature] <= t
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]
                impurity = (len(y_left) / len(y)) * self.gini(y_left) + \
                           (len(y_right) / len(y)) * self.gini(y_right)
                gain = current_impurity - impurity

                if gain > best_gain:
                    best_feat, best_thresh, best_gain = feature, t, gain
        return best_feat, best_thresh

    def build_tree(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        predicted = classes[np.argmax(counts)]

        node = {
            'leaf': True,
            'prediction': predicted
        }

        if depth < self.max_depth and len(np.unique(y)) > 1:
            feat, thr = self.best_split(X, y)
            if feat is not None:
                left_idx = X[:, feat] <= thr
                right_idx = ~left_idx
                node = {
                    'leaf': False,
                    'feature': feat,
                    'threshold': thr,
                    'left': self.build_tree(X[left_idx], y[left_idx], depth + 1),
                    'right': self.build_tree(X[right_idx], y[right_idx], depth + 1)
                }
        return node

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y, 0)

    def predict_one(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree_) for x in X])


class RandomForest:
    def __init__(self):
        self.n_estimators = 10
        self.max_depth = 5
        self.trees = []

    def bootstrap(self, X, y):
        idx = np.random.choice(len(X), len(X), replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            Xb, yb = self.bootstrap(X, y)
            tree = DecisionTree()
            tree.fit(Xb, yb)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        preds = []
        for i in range(X.shape[0]):
            vals, cnts = np.unique(tree_preds[:, i], return_counts=True)
            preds.append(vals[np.argmax(cnts)])
        return np.array(preds)


print("Decision Tree")
dt = DecisionTree()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred_dt))

print("Random Forest")
rf = RandomForest()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred_rf))
