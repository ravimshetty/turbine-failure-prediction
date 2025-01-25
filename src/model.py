from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import numpy as np

# Function to train different models
def train_model(X_train, y_train, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = SVC()
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
    
    model.fit(X_train, y_train)
    return model

# Function to optimize models using GridSearchCV
def optimize_model(X_train, y_train, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression()
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'svm':
        model = SVC()
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Return the best model
    return grid_search.best_estimator_

# Function to evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# Function to save the model
def save_model(model, filename):
    joblib.dump(model, filename)


# Function to load the model
def load_model(filename):
    return joblib.load(filename)
