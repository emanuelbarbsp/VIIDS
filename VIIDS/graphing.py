import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import itertools
import seaborn as sns

# rule of thumb: function(x_name, x_train, x_test, y_name, y_train, ytest, [all other inputs])

def actual_vs_predict_plot(x_names, x_test, y_name, y_test, predict_test, size_x=10, size_y=5, color1='blue', color2='red'):
    '''
    Displays actual vs prediction plot(s) for a given model.

        Parameters:
                x_names: List of the model's x variables
                x_test: Dictionary, keys are x axes (like Age, Model), and values are it's x values (for Age, 1, 2, 3, 4).
                y_name: String of y variable's name.
                y_test: List of actual y values.
                predict_test: Like of predicted y values.
                size_x, size_y, color1, color2: Plot dimensions and colors.
    '''

    for i in x_names:
        if i == 'intercept':
            continue
        plt.figure(figsize=(size_x, size_y))
        plt.scatter(x_test[i], y_test, color=color1, label='Actual')
        plt.scatter(x_test[i], predict_test, color=color2, label='Predicted')
        plt.xlabel(i)
        plt.ylabel(y_name)
        plt.title(f'Actual vs Predicted {y_name} ({i})')
        plt.legend()
        plt.grid(True)
        plt.show()

def gains_chart(actuals, predictions, num_buckets=10):
    '''
    Display gains chart for a models and prints the area under its curve. Could be used for train or test data.

        Parameters:
                actuals: A pandas series representing the actual results.
                predictions: A pandas series representing the model's predictions.
                num_buckets: An integer representing number of buckets.
    '''
    
    # Combine actuals and predictions into a single DataFrame
    results = np.asarray([actuals, predictions]).T
    # Sort by predictions descending
    results = results[results[:,1].argsort()[::-1]]

    # Calculate total number of instances and cumulative count
    total_count = len(results)
    cumulative_count = np.arange(1, total_count + 1) / total_count

    # Calculate cumulative actuals
    cumulative_actuals = np.cumsum(results[:,0])

    # Calculate cumulative baseline (if predictions were randomly sorted)
    baseline = np.arange(1, total_count + 1) * np.sum(actuals) / total_count

    # Calculate gains
    gains = cumulative_actuals / np.sum(actuals)

    # Calculate AUC (Area Under the Gains Curve)
    auc_score = auc(cumulative_count, gains)

    # Plotting the gains chart
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_count, gains, marker='o', linestyle='-', color='b', label=f'Gains Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Actual')
    plt.title('Gains Chart for Regression Model')
    plt.xlabel('Percentage of Population')
    plt.ylabel('Cumulative Gains')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print AUC
    print(f"Area Under the Gains Curve (AUC): {auc_score:.4f}")
    
def double_gains_plot(actuals, predicts1, predicts2, num_buckets=10):
    '''
    Display gains chart for two models and prints the area under their curves. Could be used for train or test data.

        Parameters:
                actuals: A pandas series representing the actual results.
                predicts1: A pandas series representing the first model's predictions.
                predicts2: A pandas series representing the second model's predictions.
                num_buckets: An integer representing number of buckets.
    '''
    
    # Combine actuals and predictions into a single DataFrame for each model
    results_model1 = np.asarray([actuals, predicts1]).T
    results_model2 = np.asarray([actuals, predicts2]).T
    
    # Sort by predictions descending for each model
    results_model1 = results_model1[results_model1[:,1].argsort()[::-1]]
    results_model2 = results_model2[results_model2[:,1].argsort()[::-1]]

    # Calculate total number of instances and cumulative count
    total_count = len(results_model1)
    cumulative_count = np.arange(1, total_count + 1) / total_count

    # Calculate cumulative actuals for each model
    cumulative_actuals_model1 = np.cumsum(results_model1[:,0])
    cumulative_actuals_model2 = np.cumsum(results_model2[:,0])

    # Calculate cumulative baseline (if predictions were randomly sorted)
    baseline = np.arange(1, total_count + 1) * np.sum(actuals) / total_count

    # Calculate gains for each model
    gains_model1 = cumulative_actuals_model1 / np.sum(actuals)
    gains_model2 = cumulative_actuals_model2 / np.sum(actuals)

    # Calculate AUC (Area Under the Gains Curve) for each model
    auc_score_model1 = auc(cumulative_count, gains_model1)
    auc_score_model2 = auc(cumulative_count, gains_model2)

    # Plotting the gains chart for both models
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_count, gains_model1, marker='o', linestyle='-', color='b', label=f'{predicts1.name} (AUC = {auc_score_model1:.2f})')
    plt.plot(cumulative_count, gains_model2, marker='s', linestyle='-', color='g', label=f'{predicts2.name} (AUC = {auc_score_model2:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Baseline')
    plt.title(f'Gains Chart for {predicts1.name} and {predicts2.name}')
    plt.xlabel('Percentage of Population')
    plt.ylabel('Cumulative Gains')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print AUC for each model
    print(f"{predicts1.name} - Area Under the Gains Curve (AUC): {auc_score_model1:.4f}")
    print(f"{predicts2.name} - Area Under the Gains Curve (AUC): {auc_score_model2:.4f}")

def lift_chart(actuals, predictions, num_buckets=10):
    '''
    Display lift chart for a model. Could be used for train or test data.

        Parameters:
                actuals: A pandas series representing the actual results.
                predictions: A pandas series representing the model's predictions.
                num_buckets: An integer representing number of buckets.
    '''
    # Create helper function for displaying lift charts
    # Combine actuals and predictions into a single DataFrame
    results = np.asarray([actuals, predictions]).T
    # Sort by predictions descending
    results = results[results[:,1].argsort()[::-1]]

    # Calculate total number of instances and cumulative count
    total_count = len(results)
    cumulative_count = np.arange(1, total_count + 1) / total_count

    # Calculate cumulative actuals
    cumulative_actuals = np.cumsum(results[:,0])

    # Calculate cumulative baseline (if predictions were randomly sorted)
    baseline = np.arange(1, total_count + 1) * np.sum(actuals) / total_count

    # Calculate lift
    lift = cumulative_actuals / baseline

    # Calculate cumulative lift
    cumulative_lift = np.cumsum(lift)

    # Plotting the lift chart
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_count, cumulative_lift, marker='o', linestyle='-', color='b', label='Lift Curve')
    plt.plot([0, 1], [1, 1], linestyle='--', color='r', label='Baseline')
    plt.title('Lift Chart for Regression Model')
    plt.xlabel('Percentage of Population')
    plt.ylabel('Lift')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def importance_plot(imp_series, figsize=(8, 6)):
    """
    Displays a plot of the x variables and their importances.
        
        Parameters:
            x_names: List of x variable names, could include "intercept". Usually obtained by modeldata.columns
            imp_series: A series of variables and their importances.
            figsize: Tuple of plot dimensions.
    """

    x_names = imp_series.index
    importances = []
    for name in x_names:
        if name == 'intercept':
            continue
        importances.append(imp_series[name])
    
    # Sort variables and importances in descending order of importances
    indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
    sorted_variables = [x_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(importances)), sorted_importances, align='center')
    plt.xticks(range(len(importances)), sorted_variables, rotation=90)
    plt.xlabel('Variable')
    plt.ylabel('Importance')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.show()

def histogram_plot(df):
    """
    Displays histogram plots for numerical x variables of a given pandas dataframe.

        Parameters:
            df: The pandas dataframe to be plotted.
    """

    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for x in numerical_cols:
        df[x].hist(bins=50)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {x}')
        plt.show()

def density_plot(df):
    """
    Displays density plots for numerical x variables of a given pandas dataframe.

        Parameters:
            df: The pandas dataframe to be plotted.
    """

    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for x in numerical_cols:
        df[x].plot(kind='density')
        plt.xlabel('Value')
        plt.title(f'Density Plot of {x}')
        plt.show()

def box_plot(df):
    """
    Displays box plots for numerical x variables of a given pandas dataframe.

        Parameters:
            df: The pandas dataframe to be plotted.
    """
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for x in numerical_cols:
        df.boxplot(column=x)
        plt.title(f'Box Plot of {x}')
        plt.show()

def frequency_table(df):
    """
    Prints frequency tables for numerical x variables of a given pandas dataframe.

        Parameters:
            df: The pandas dataframe to be plotted.
    """

    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for x in numerical_cols:
        frequency_table = df[x].value_counts()
        print(frequency_table)
        
def scatter_matrix_plot(df):
    """
    Displays scatter plots for numerical x variables of a given pandas dataframe.

        Parameters:
            df: The pandas dataframe to be plotted.
    """
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Generate scatter plots for every pair of numerical columns
    for (col1, col2) in itertools.combinations(numerical_cols, 2):
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col1], df[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter Plot of {col1} vs {col2}')
        plt.grid(True)
        plt.show()

def correlation_coefficient_plot(df):
    """
    Displays the correlation coefficients for numerical variables of a given pandas dataframe.

    Parameters:
        df: The pandas dataframe to be analyzed.
    """
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize a dictionary to store correlation coefficients
    correlations = {}
    
    # Compute correlation coefficients for every pair of numerical columns
    for col1, col2 in itertools.combinations(numerical_cols, 2):
        correlation = df[col1].corr(df[col2])
        correlations[(col1, col2)] = correlation
    
    # Print correlation coefficients
    for (col1, col2), corr in correlations.items():
        print(f'Correlation between {col1} and {col2}: {corr:.2f}')

def correlation_heatmap(df):
    """
    Displays a heatmap of the correlation matrix for numerical variables of a given pandas dataframe.

    Parameters:
        df: The pandas dataframe to be analyzed.
    """
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Compute the correlation matrix
    correlation_matrix = df[numerical_cols].corr()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
                linewidths=0.5, square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def joint_plot(df):
    """
    Displays joint plots for numerical variables of a given pandas dataframe.

    Parameters:
        df: The pandas dataframe to be plotted.
    """
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Generate joint plots for every pair of numerical columns
    for (col1, col2) in itertools.combinations(numerical_cols, 2):
        sns.jointplot(x=col1, y=col2, data=df, kind='scatter')
        plt.show()
