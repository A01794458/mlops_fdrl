from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

import yaml

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

    
class Preprocessor:
    def scale_features(self, data, numeric_columns):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return data

    def encode_categorical_columns(self, data, categorical_columns):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])
        return data

    def apply_pca(self, data, numeric_columns, n_components=2):
        if isinstance(data, int):
            raise ValueError("El parámetro 'data' no puede ser de tipo 'int'.")
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data[numeric_columns])
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        return data

    def preprocess_data(self, df, numeric_columns, categorical_columns):
        if isinstance(df, int):
            raise ValueError("El parámetro 'df' no puede ser de tipo 'int'.")
        
        df = self.scale_features(df, numeric_columns)
        df = self.encode_categorical_columns(df, categorical_columns)
        df = self.apply_pca(df, numeric_columns)
        return df