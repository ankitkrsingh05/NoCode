import pandas as pd
import numpy as np
from scipy import stats
from modules.visualization import add_bell_curve,plot_distribution,plot_qq_plot
def perform_normality_tests(data, column):
    # Shapiro-Wilk test
    _, shapiro_p_value = stats.shapiro(data[column])
    print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
    
    # Agostino test
    _, agostino_p_value = stats.normaltest(data[column])
    print(f"Agostino test p-value: {agostino_p_value}")
    
    # Anderson-Darling test
    anderson_statistic, anderson_critical_values, anderson_significance = stats.anderson(data[column], dist='norm')
    print(f"Anderson-Darling test statistic: {anderson_statistic}")
    print(f"Anderson-Darling test critical values: {anderson_critical_values}")

def apply_transformation(df, columns, transformation):
    transformed_df = df.copy()
    
    for column in columns:
        # Plot before transformation
        plot_qq_plot(df[column], f"QQ Plot - Before Transformation: {column}")
        plot_distribution(df[column], f"Distribution Plot - Before Transformation: {column}")
        add_bell_curve(df, column)
        
        # Perform normality tests before transformation
        perform_normality_tests(df, column)
        
        # Apply transformation
        if transformation == 'log':
            transformed_df[column] = np.log(df[column])
        elif transformation == 'log1p':
            transformed_df[column] = np.log1p(df[column])
        elif transformation == 'square':
            transformed_df[column] = np.square(df[column])
        elif transformation == 'sqrt':
            transformed_df[column] = np.sqrt(df[column])
        elif transformation == 'reciprocal':
            transformed_df[column] = np.reciprocal(df[column])
        elif transformation == 'boxcox':
            transformed_df[column], _ = stats.boxcox(df[column])
        elif transformation == 'yeo-johnson':
            transformed_df[column], _ = stats.yeojohnson(df[column])
        else:
            print(f"Unsupported transformation: {transformation}")
    
        # Plot after transformation
        plot_qq_plot(transformed_df[column], f"QQ Plot - After Transformation: {column}")
        plot_distribution(transformed_df[column], f"Distribution Plot - After Transformation: {column}")
        add_bell_curve(transformed_df, column)
        
        # Perform normality tests after transformation
        perform_normality_tests(transformed_df, column)
        
        print()
    
    return transformed_df
