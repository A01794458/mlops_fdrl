import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yaml

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

class DataHandler:
    @staticmethod
    def load_data(filepath, sep=";"):
        if isinstance(filepath, int):
            raise ValueError("El parámetro 'filepath no puede ser de tipo 'int'.")
        
        print("1.- Loading and exploring the data")
        data = pd.read_csv(filepath, sep=sep)
        return data

    @staticmethod
    def prepared_data(data):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        print("2.- Prepared the data")
        print(data.describe())
        print("Número de filas en el dataset:", data.shape[0])
        print("Número de columnas en el dataset:", data.shape[1])
        print("\nPrimeras filas del conjunto de datos:")
        print(data.head())

        # Revisar valores nulos
        print("\nRevisar valores nulos:")
        missing_data_summary = data.isnull().sum()
        print(missing_data_summary)

        # Eliminar filas con valores nulos
        df_cleaned = data.dropna()
        print("\nValores nulos después de limpieza:")
        print(df_cleaned.isnull().sum())

        # Resumen del DataFrame limpio
        print("\nResumen del DataFrame limpio:")
        print(df_cleaned.info())

        f = open( 'reports/data/prepared_data.txt', 'w' )
        f.write( "2.- Prepared the data\n")
        f.write(str(data.describe()))
        f.write('\n')
        f.write("Número de filas en el dataset: " + str(data.shape[0]) + '\n')
        f.write("Número de columnas en el dataset: " + str(data.shape[1]) + '\n')
        f.write("\nPrimeras filas del conjunto de datos:\n")
        f.write(str(data.head()))
        f.write('\n')
        f.write("\nRevisar valores nulos:\n")
        f.write(str(missing_data_summary))
        f.write("\nValores nulos después de limpieza:\n")
        f.write(str(df_cleaned.isnull().sum()))
        print("\nResumen del DataFrame limpio:\n")
        print(str(df_cleaned.info()))
        f.close()

        return df_cleaned

    @staticmethod
    def plot_histograms(data):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        x = data.hist(bins=15, figsize=(15, 10))
        y = sns.histplot(data,bins=15)
        y.title = 'Histograma'
        plt.savefig('./reports/data/histogram.png')
        
        

    @staticmethod
    def plot_correlation_matrix(data):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        plt.figure(figsize=(12, 8))
        plt.title('Matriz de correlaciones')
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.savefig('./reports/data/correlation.png')
        plt.close()
        #plt.show()

    @staticmethod
    def plot_feature_relationships(data, target):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data)
        plt.xticks(rotation=90)
        plt.title(f'Diagrama de caja de todas las variables numéricas con respecto a {target}')
        plt.savefig('./reports/data/feature_relationships.png')
        #plt.show()
    
    @staticmethod
    def versioned_data(data):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        print("3.- Versioned the data")
        data.to_csv(config['data']['processed_data_route']+config['data']['processed_data_csv']+'_'+config['data']['version']+'.csv', index=False)
        print("Versión de datos: " + config['data']['version'])
        print("Hash del archivo:", hash(tuple(data.values.tobytes())))