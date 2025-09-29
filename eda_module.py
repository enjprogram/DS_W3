#EDA module

# Импорт стандартных модулей
import sys
import collections
from typing import List, Tuple

# Импорт сторонних библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.plotting import figure, show, output_file
from bokeh.io import output_file
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource

from bokeh.models import ColumnDataSource, LabelSet
from sklearn.linear_model import LinearRegression

import decor_module as d

#----------------------------------Show Types and Stats------------------------------------------------
@d.data_descr_decorator
def view_data(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.dtypes)
    return df


#-----------------------------------Histograms---------------------------------------------------------

@d.hist_decorator
def plot_histogram(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn для красивых графиков
    sns.set(style="whitegrid")
    
    # Создание гистограмм для каждой числовой переменной
    df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    
    # Добавление названий для каждого графика и осей
    for ax in plt.gcf().get_axes():
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')
        ax.set_title(ax.get_title())
        #ax.set_title(ax.get_title().replace('wine_class', 'Класс вина'))
    
    # Регулировка макета для предотвращения наложения подписей
    plt.tight_layout()
    
    # Показать график
    plt.show()


#-------------------------------------Heatmaps----------------------------------------------------------
@d.heatmap_decorator
def plot_heatmap(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="white")
    
    # Расчет корреляционной матрицы только для числовых данных
    numeric_df = df.select_dtypes(include=[np.number])  # Исключаем нечисловые столбцы
    corr = numeric_df.corr()
    
    # Маска для отображения только нижней треугольной части матрицы (опционально)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Настройка цветовой палитры
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Создание тепловой карты
    plt.figure(figsize=(30, 16))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    # Добавление заголовка
    plt.title('Тепловая карта корреляций', fontsize=20)
    
    # Показать график
    plt.show()


#------------------------------------WhiskerBox-----------------------------------------------------------
@d.whisker_box_decorator
def plot_whisker_box(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="whitegrid")
    
    # Предполагаем, что df — это ваш DataFrame
    # Создаем ящики с усами для каждой колонки в DataFrame
    plt.figure(figsize=(12, 50))
    
    # Перебираем каждый числовой столбец и создаем для него ящик с усами
    for index, column in enumerate(df.select_dtypes(include=[np.number]).columns):
        plt.subplot((len(df.columns) // 3) + 1, 3, index + 1)
        sns.boxplot(y=df[column])
    
    plt.tight_layout()
    plt.show()

#---------------------------------Bokeh Bar Plot---------------------------------------------------------
def plot_bar_plot(df, target_column, features, output_filename='feature_distribution_by_survival_rate_bar_plot.html'):
    output_file(output_filename)
    output_notebook()
    plots = []

    # Define colors for survival status 0 and 1
    survival_colors = [Category10[10][0], Category10[10][1]]  # Use the first two colors from Category10 palette

    for feature in features:
        # Create the frequency table for each feature and survival rate
        feature_count = df.groupby([target_column, feature]).size().unstack(fill_value=0)

        # Ensure columns 0 and 1 exist for the survival rates
        if 0 not in feature_count.columns:
            feature_count[0] = 0
        if 1 not in feature_count.columns:
            feature_count[1] = 0

        survival_status_values = list(feature_count.index)  # Rows for survival status (0.0, 1.0)
    
        # Handle the categorical bins for features (X-axis)
        x_range = [str(i) for i in feature_count.columns]

        # Initialize the plot
        p = figure(x_range=x_range, title=f'{feature} distribution by {target_column}', toolbar_location=None, tools="", height=350, width=350)

        # Iterate through the survival status (0 and 1) to create bars
        for i, survival_status in enumerate(survival_status_values):
        
            top_values = feature_count.loc[survival_status, :].values

            color = survival_colors[i] 

            # Make sure the x_range and top_values align properly
            p.vbar(x=x_range, top=top_values, width=0.4, 
                   color=color,  # Corrected color indexing
                   legend_label=f'{target_column} {int(survival_status)}')

        # Set axis labels and legend
        p.xaxis.axis_label = feature
        p.yaxis.axis_label = "Count"
        p.legend.title = f"{target_column}"
        p.legend.location = "top_left"

        # Append the plot to the list of plots
        plots.append(p)

    # Arrange the plots in a grid (3 per row)
    grid = gridplot([plots[i:i+3] for i in range(0, len(plots), 3)])
    show(grid)

#---------------------------------Bokeh Scatter Linear Regression Plots-----------------------------------
def plot_scatter_with_regression(df, target_column, features, output_filename='scatter_with_regression_by_survival_rate.html'):
    output_file(output_filename)
    output_notebook()
    plots = []

    # Define colors for survival status 0 and 1
    survival_colors = [Category10[10][0], Category10[10][1]]

    for feature in features:
        # Filter data for survival status 0 and 1
        survival_0 = df[df[target_column] == 0]
        survival_1 = df[df[target_column] == 1]

        # Create a ColumnDataSource for each survival status
        source_0 = ColumnDataSource(data={'x': survival_0[feature], 'y': np.zeros(len(survival_0))})
        source_1 = ColumnDataSource(data={'x': survival_1[feature], 'y': np.ones(len(survival_1))})

        # Create scatter plot
        p = figure(title=f'{feature} vs {target_column}', width=350, height=350)
        p.scatter(x='x', y='y', source=source_0, color=survival_colors[0], legend_label='Survival 0', size=8)
        p.scatter(x='x', y='y', source=source_1, color=survival_colors[1], legend_label='Survival 1', size=8)

        # Add a regression line
        X = df[[feature]].dropna() 
        y = df[target_column].dropna()
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        p.line(X[feature], y_pred, line_color='gray', legend_label='Regression Line', line_width=2)

        p.xaxis.axis_label = feature
        p.yaxis.axis_label = target_column
        p.legend.title = f"{target_column}"

        plots.append(p)

    grid = gridplot([plots[i:i + 3] for i in range(0, len(plots), 3)])
    show(grid)


#-------------------------Density Plots-----------------------------------------------------------
# Function to plot KDE using Seaborn and Bokeh
def plot_density(df, target_column, features, output_filename='density_plot_by_survival_rate.html'):
    output_file(output_filename, title="Density Plot by Survival Status")
    output_notebook()
    plots = []
    
    # Define colors for survival status 0 and 1
    survival_colors = [Category10[10][0], Category10[10][1]]

    for feature in features:
        # Separate the data into two groups based on survival status
        survival_0_data = df[df[target_column] == 0][feature].dropna()
        survival_1_data = df[df[target_column] == 1][feature].dropna()
        
        # Create a Seaborn KDE plot for each group
        plt.figure(figsize=(7, 5))
        sns.kdeplot(survival_0_data, color=survival_colors[0], label="Survival 0", fill=True)
        sns.kdeplot(survival_1_data, color=survival_colors[1], label="Survival 1", fill=True)

        plt.title(f'Density Plot of {feature} by {target_column}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()

        # Save the plot as an image
        plt.tight_layout()
        plt.savefig(f'{feature}_kde_plot.png')
        plt.close()

        # Display in Bokeh
        p = figure(title=f'Density Plot for {feature}', x_axis_label=feature, y_axis_label='Density', width=500, height=400)
        
        # To add an image as background
        img_path = f'{feature}_kde_plot.png'
        p.image_url(url=[img_path], x=0, y=1, w=1, h=1)
        plots.append(p)

    grid = gridplot([plots[i:i + 2] for i in range(0, len(plots), 2)])
    show(grid)


#------------------Box Density Plots---------------------------------------------------------------------

def plot_bokeh_density(df, target_column, features, output_filename='bokeh_density_plot_by_survival_rate.html'):
    output_file(output_filename, title="Density Plot by Survival Status")
    output_notebook()
    plots = []

    # Define colors for survival status 0 and 1
    survival_colors = [Category10[10][0], Category10[10][1]]

    for feature in features:
        # Get the data for survival status 0 and 1
        survival_0_data = df[df[target_column] == 0][feature].dropna()
        survival_1_data = df[df[target_column] == 1][feature].dropna()

        # Create histograms and calculate density values
        hist_0, edges_0 = np.histogram(survival_0_data, bins=30, density=True)
        hist_1, edges_1 = np.histogram(survival_1_data, bins=30, density=True)

        # Create the x values for density (step function)
        x_0 = 0.5 * (edges_0[1:] + edges_0[:-1])  # Midpoint of bin edges
        x_1 = 0.5 * (edges_1[1:] + edges_1[:-1])  # Midpoint of bin edges

        # Create the figure
        p = figure(title=f'Density Plot of {feature} by {target_column}', 
                   x_axis_label=feature, y_axis_label='Density', width=500, height=350)

        # Plot using step
        p.step(x_0, hist_0, line_width=2, line_color=survival_colors[0], line_alpha=0.6, legend_label="Survival 0")
        p.step(x_1, hist_1, line_width=2, line_color=survival_colors[1], line_alpha=0.6, legend_label="Survival 1")

        p.legend.title = "Survival Status"
        p.legend.location = "top_right"

        plots.append(p)

    from bokeh.layouts import gridplot
    grid = gridplot([plots[i:i + 2] for i in range(0, len(plots), 2)])
    show(grid)