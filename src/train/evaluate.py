import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
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

from src.train.training import ModelTraining

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

class ModelEvaluation:
    def train_and_evaluate_models(self, models, param_grids, X_train, y_train, X_test, y_test):
        if isinstance(models, int):
            raise ValueError("El parámetro 'models' no puede ser de tipo 'int'.")

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            # Configuración de MLflow para registrar el experimento
            mlflow.set_tracking_uri(config['train']['tracking_uri'])
            mlflow.set_experiment(f"ML_Models_Absenteeism_Equipo4_Fase4_{model_name}")

            with mlflow.start_run(nested=True) as run:
                # Registrar hiperparámetros en MLflow
                mlflow.log_param("model_name", model_name)

                # Crear pipeline del modelo
                model_pipeline = Pipeline(steps=[('classifier', model)])
                best_model = ModelTraining().get_best_model(param_grids[model_name], model_pipeline, X_train, y_train, model_name)

                print('_______*************' + model_name + '****************____-')
                # Evaluar el modelo y registrar métricas en MLflow
                accuracy, report, cm, y_pred = ModelTraining().evaluate_classification_model(best_model, X_test, y_test)
                
                # Calcular precisión y recuperación
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                # Registrar métricas en MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                print(f"{model_name} Accuracy: {accuracy}\n")
                print(f"Classification Report for {model_name}:\n{report}\n")
                ModelTraining().plot_confusion_matrix(cm, model_name)

                # Registrar el modelo en MLflow
                mlflow.sklearn.log_model(best_model, "model")
                print(f"Modelo registrado en MLflow para {model_name}")

                # Validación cruzada
                cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=2), scoring='accuracy')
                mlflow.log_metric("cross_val_accuracy", np.mean(cross_val_scores))
                print(f"{model_name} Cross-Validation Accuracy: {np.mean(cross_val_scores)}\n")

                # Importancia de características para Random Forest y Gradient Boosting
                if model_name in ['Random Forest', 'Gradient Boosting']:
                    ModelTraining().plot_feature_importance(best_model.named_steps['classifier'], X_train.columns, model_name)

    
    def train_and_evaluate_models_with_balancing(self, models, param_grids, X_train, y_train, X_test, y_test):
        
        if isinstance(models, int):
            raise ValueError("El parámetro 'models' no puede ser de tipo 'int'.")
        
        # Aplicar sobremuestreo a los datos de entrenamiento
        print("\nAplicando RandomOverSampler para balanceo de clases...")
        ros = RandomOverSampler(random_state=config['base']['random_state'])
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
        print(f"Nuevas formas de X_train: {X_train_resampled.shape}, y_train: {y_train_resampled.shape}")
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name} con datos balanceados...")

            # Configuración de MLflow para registrar el experimento
            mlflow.set_tracking_uri(config['train']['tracking_uri'])
            mlflow.set_experiment(f"ML_Models_Absenteeism_Equipo4_Fase4_Balanced_{model_name}")

            with mlflow.start_run(nested=True) as run:
                # Registrar hiperparámetros en MLflow
                mlflow.log_param("model_name", model_name)

                # Crear pipeline del modelo
                model_pipeline = Pipeline(steps=[('classifier', model)])
                best_model = ModelTraining().get_best_model(param_grids[model_name], model_pipeline, X_train_resampled, y_train_resampled, model_name)
                print('*************' + model_name + '****************')
                # Evaluar el modelo y registrar métricas en MLflow
                accuracy, report, cm, y_pred = ModelTraining().evaluate_classification_model(best_model, X_test, y_test)
                
                # Calcular precisión y recuperación
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                # Registrar métricas en MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                print(f"{model_name} Accuracy: {accuracy}\n")
                print(f"Classification Report for {model_name}:\n{report}\n")
                ModelTraining().plot_confusion_matrix(cm, model_name)

                # Registrar el modelo en MLflow
                mlflow.sklearn.log_model(best_model, "model")
                print(f"Modelo registrado en MLflow para {model_name}")

                # Validación cruzada
                cross_val_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=StratifiedKFold(n_splits=2), scoring='accuracy')
                mlflow.log_metric("cross_val_accuracy", np.mean(cross_val_scores))
                print(f"{model_name} Cross-Validation Accuracy: {np.mean(cross_val_scores)}\n")

                # Importancia de características para Random Forest y Gradient Boosting
                if model_name in ['Random Forest', 'Gradient Boosting']:
                    ModelTraining().plot_feature_importance(best_model.named_steps['classifier'], X_train.columns, model_name)             

