#load_convert_module
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.io import arff

import eda_module as eda
import decor_module as d

def load_data(file_path):
    """
    Загрузка данных из CSV файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found at {file_path}')

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_index:
                zip_files = zip_index.namelist()
                print(f'Files in zip {zip_files}')
                zip_file = zip_files[0]
                with zip_index.open(zip_file) as file:
                    return _load_by_extension(file_path, zip_file)
                    
        file_ext = os.path.splitext(file_path)[1].lower()

        return _load_by_extension(file_path, file_ext)

    except FileNotFoundError as e:
        print(f"File not found: {e}")

    except ValueError as e:
        print(f'Value error: {e}')

    except Exception as e:
        print(f'Undefined error: {e}')

    finally:
        print(f'Attempted data loading from file at {file_path}')


def _load_by_extension(file_path, file_ext):
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)

    elif file_ext == '.arff':
        data, meta = arff.loadarff(file_path)
        return pd.DataFrame(data)

    elif file_ext == '.json':
        return pd.read_json(file_path)

    elif file_ext == '.txt':
        return pd.read_txt(file_path, delimiter = '\t') # tab separation is assumed

    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)

    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)

    else:
        raise ValueError(f'Unsupported file {file_ext}')

def decode_bytes_to_int(value):
    if isinstance(value, bytes):
        try:
            decoded = value.decode('utf-8')
            if decoded == '?':
                return np.nan
            return int(decoded)
        except ValueError:
            return np.nan
    return value

def convert_df_bytes_to_int(df):
    return df.map(decode_bytes_to_int)


# def convert_query_response(df):
#     for col in df.columns:
#         df[col]=df[col].apply(decode_bytes_to_int)
#     return df