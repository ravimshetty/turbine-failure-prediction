# model.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Define the models
def train_model(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }
    
    # Train each model and store it in a dictionary
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    
    return trained_models


def evaluate_model(models, X_test, y_test):
    evaluation_results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        evaluation_results[model_name] = classification_report(y_test, y_pred)
    
    return evaluation_results


def optimize_model(models, X_train, y_train):
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'DecisionTree': {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    }
    
    # Perform grid search for optimization
    optimized_models = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        optimized_models[model_name] = grid_search.best_estimator_

    return optimized_models


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)
