# ml_module.py
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


def clean_data(df, drop_columns):
    
    df = df.drop(columns = drop_columns) 
    df = df.dropna() # remove nan values
    df = df.drop_duplicates() # remove duplicates

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df)
    return df
    
def preprocess_data(df, drop_columns, target_column):
    """
    Предобработка данных: разделение на признаки и целевую переменную, масштабирование признаков.
    :param df: DataFrame с данными.
    :param target_column: Имя столбца с целевой переменной.
    :return: Обработанные признаки, целевая переменная, препроцессор.
    """

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
        
    df = df.drop(columns = drop_columns) 
    df = df.dropna() # remove nan values
    df = df.drop_duplicates() # remove duplicates

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df)

    # imp = SimpleImputer(strategy = "most_frequent" )
    # imp.fit_transform(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Применение препроцессора к данным
    X_processed = X
    print("Данные успешно предобработаны.")
    
    return X_processed, y

def train_model(X, y):
    """
    Обучение модели линейной регрессии.
    :param X: Признаки.
    :param y: Целевая переменная.
    :return: Обученная модель.
    """
    """
    Обучение модели на обучающих данных.
    """
    if X is None or y is None:
        print("Данные не загружены или не предобработаны.")
        return

    try:
        model = LinearRegression()
        model.fit(X, y)
        print("Модель успешно обучена.")
        return model
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

def predict(model, X):
    """
    Предсказание на новых данных.
    :param model: Обученная модель.
    :param X: Признаки.
    :return: Предсказанные значения.
    """
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    """
    Оценка модели с использованием метрик MSE и R^2.
    :param y_true: Истинные значения.
    :param y_pred: Предсказанные значения.
    :return: MSE, R^2.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

