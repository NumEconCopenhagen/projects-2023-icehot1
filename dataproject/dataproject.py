# my_functions.py
import pandas as pd
import statsmodels.api as sm

def run_regression(dfs, categories, columns, independent_vars):
    # List to hold the results
    results = []

    # Calculate stats for each category
    for category in categories:
        for column in columns:
            # Define dependent and independent variables
            X = dfs[category][independent_vars]
            y = dfs[category][column]

            # Add state fixed effects
            X = pd.concat([X, pd.get_dummies(dfs[category]['State'], drop_first=True)], axis=1)

            # Add constant to independent variables
            X = sm.add_constant(X)

            # Create OLS regression model
            model = sm.OLS(y, X, missing='drop')
            result = model.fit()

            # Check if the share_black2000 coefficient is statistically significant
            coef = result.params['share_black2000']
            pvalue = result.pvalues['share_black2000']
            if pvalue < 0.05:
                coef_str = "{:.4f}*".format(coef)
            else:
                coef_str = "{:.4f}".format(coef)

            # Append the coefficient of share_black2000 to the list
            results.append([category, column, coef_str])

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results, columns=['Category', 'Variable', 'Coefficient'])

    # Print the DataFrame
    print(df_results)