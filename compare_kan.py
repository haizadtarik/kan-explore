import imodelsx
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_iris()
X = data.data
y = data.target
model = imodelsx.KANClassifier(hidden_layer_sizes=[32, 64], device='cuda',
                               regularize_activation=1.0, regularize_entropy=1.0)
model.fit(X, y)
y_pred = model.predict(X)
print('KAN Test acc', accuracy_score(y, y_pred))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
y_pred_rf = clf.predict(X)
print('RF Test acc', accuracy_score(y, y_pred_rf))


mlp = MLPClassifier(random_state=1, max_iter=300)
mlp.fit(X, y)
y_pred_mlp = mlp.predict(X)
print('MLP Test acc', accuracy_score(y, y_pred_mlp))