from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits

mnist=load_digits()
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
clf = SVC(kernel = 'rbf')
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
tree_param_grid = { 'kernel': ['rbf', 'linear', 'sigmoid', 'poly'], 'C': [0.5, 0.6, 0.8, 1.0, 2.0]
                   ,'gamma': ['auto', 0.001]}
grid = GridSearchCV(SVC(max_iter = -1),param_grid=tree_param_grid, scoring = 'accuracy', cv = 5)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
clf_best = grid.best_estimator_
print(accuracy_score(y_test, clf_best.predict(X_test)))