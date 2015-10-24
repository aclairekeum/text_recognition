from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import glob
import os

def dump_model(model):
    files = glob.glob('model/*')  # Clear out the old files in model/
    for f in files:
        os.remove(f)
    joblib.dump(model, 'model/model.pkl')  # Dump model

def logistic_regression():
    tuned_parameters = [{'C': [1e-3, 1e-2, 1e-1, 1e0]}]
    return GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, n_jobs=-1)

def support_vector_machine():
    tuned_parameters = [{'kernel': ['rbf'],'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    return GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=-1)

def k_nearest_neighbors():
    tuned_parameters = [{'weights': ['uniform'], 'n_neighbors': range(1,11)},
                        {'weights': ['distance'], 'n_neighbors': range(1,11)}]
    return GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, n_jobs=-1)

def random_forest():
    tuned_parameters = [{'n_estimators': range(5, 25, 2)}]
    return GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, n_jobs=-1)

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.9)

# model = logistic_regression()
# model = support_vector_machine()
model = k_nearest_neighbors()
# model = random_forest()
# model.fit(X_train, y_train)
model.fit(data.data, data.target)

print model.best_estimator_
print model.score(data.data, data.target)
print model.score(X_train, y_train)
print model.score(X_test, y_test)

dump_model(model)
