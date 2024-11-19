import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,precision_score, recall_score,mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV

import yaml

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

class ModelTraining:
    def define_models(self):
        return {
            'Logistic Regression': LogisticRegression(max_iter=config['base']['max_iter'], random_state=config['base']['random_state']),
            'Random Forest': RandomForestClassifier(random_state=config['base']['random_state'], n_estimators=config['base']['n_estimators'], max_depth=None),
            'Gradient Boosting': GradientBoostingClassifier(random_state=config['base']['random_state'], n_estimators=config['base']['n_estimators'], max_depth=config['base']['max_depth']),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=config['base']['n_neighbors'])
        }
    
    def define_hyperparameters(self):  # Agregar self aquí
        return {
            'Logistic Regression': {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            'Random Forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5]
            },
            'K-Nearest Neighbors': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance']
            }
        }

    
    def split_data(self, data, target, test_size=config['data']['test_size'], random_state=config['base']['random_state']):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        X = data.drop(target, axis=1)
        y = data[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def get_best_model(self, param_grid, model_pipeline, X_train, y_train, model_name):
        if isinstance(model_name, int):
            raise ValueError("El parámetro 'model_name' no puede ser de tipo 'int'.")
        
        grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=StratifiedKFold(n_splits=2), scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_classification_model(self, model, X_test, y_test):
        if isinstance(model, int):
            raise ValueError("El parámetro 'model' no puede ser de tipo 'int'.")
        
        print("")
        
        with open("absenteeism_model.pkl", "wb") as f:
            pickle.dump(model, f)
        #print(f"Model saved as 'model.pkl'")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, report, cm, y_pred

    def plot_confusion_matrix(self, cm, model_name):
        if isinstance(model_name, int):
            raise ValueError("El parámetro 'model_name' no puede ser de tipo 'int'.")
        
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Etiqueta verdadera')
        plt.xlabel('Etiqueta predicha')
        plt.savefig('./reports/train/confusion_' + model_name +'.png')
        #plt.show()
        

    def plot_feature_importance(self, model, feature_names, model_name):
        if isinstance(model, int):
            raise ValueError("El parámetro 'model' no puede ser de tipo 'int'.")
        
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color="b")
        plt.title(f'Importancia de características - {model_name}')
        plt.savefig('./reports/train/characteristics_' + model_name +'.png')
        #plt.show()