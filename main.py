
import mlflow


import yaml

from src.data.handler import DataHandler
from src.data.preprocess import Preprocessor
from src.train.training import ModelTraining
from src.train.evaluate import ModelEvaluation

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

#mlflow
#zsh mlflow server --host 127.0.0.1 --port 5000

mlflow.set_tracking_uri(config['train']['tracking_uri'])
mlflow.set_experiment(config['train']['experiment'])

filepath=config['data']['raw_data_csv']
# 1. Cargar y preparar los datos
data = DataHandler.load_data(filepath)
df_cleaned = DataHandler.prepared_data(data)

# 2. Exploración y preprocesamiento de datos
DataHandler.plot_histograms(df_cleaned)
DataHandler.plot_correlation_matrix(df_cleaned)
DataHandler.plot_feature_relationships(df_cleaned, 'Absenteeism time in hours')

# Definir las columnas numéricas y categóricas
numeric_columns = ['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target']
categorical_columns = ['Month of absence', 'Day of the week', 'Seasons', 'Education', 'Disciplinary failure', 'Social drinker', 'Social smoker']

# Preprocesamiento de los datos
preprocessor = Preprocessor()
df_preprocessed = preprocessor.preprocess_data(df_cleaned, numeric_columns, categorical_columns)


# 4. Dividir los datos para entrenamiento y evaluación
model_training = ModelTraining()
X_train, X_test, y_train, y_test = model_training.split_data(df_preprocessed, 'Absenteeism time in hours')

models = model_training.define_models()
param_grids = model_training.define_hyperparameters()

print("\n### Entrenamiento y Evaluación Inicial ###")
model_evaluation = ModelEvaluation()
model_evaluation.train_and_evaluate_models(models, param_grids, X_train, y_train, X_test, y_test)

# 5. Aplicar Mejores Prácticas en el Pipeline de Modelado (balanceo de clases y optimización de hiperparámetros)
print("\n### Aplicación de Mejores Prácticas ###")
model_evaluation.train_and_evaluate_models_with_balancing(models, param_grids, X_train, y_train, X_test, y_test)
