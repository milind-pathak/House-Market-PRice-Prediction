import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the dataset
df = pd.read_csv('/Users/YouShallNotPass/Desktop/BD with ML Project/Model-Train/House-Data-Clean.csv')

# 1. Remove empty columns
df.dropna(axis=1, how='all', inplace=True)
# 2. Remove rows with any empty cells
df.dropna(axis=0, how='any', inplace=True)

# Convert 'period_begin' to datetime and extract 'year' and 'month'
df['period_begin'] = pd.to_datetime(df['period_begin'])
df['year'] = df['period_begin'].dt.year
df['month'] = df['period_begin'].dt.month

# Fill NaN values with median
#df.fillna(df.median(), inplace=True)

num_rows = df.shape[0]
print("Number of rows:", num_rows)

df = df[~df['year'].isin([2015, 2016, 2017])]

df.reset_index(drop=True, inplace=True)

num_rows = df.shape[0]
print("Number of rows after removing years:", num_rows)

rows_2012 = df[df['year'] == 2012]
print("Number of rows for year 2012:", len(rows_2012))
rows_2013 = df[df['year'] == 2013]
print("Number of rows for year 2013:", len(rows_2013))
rows_2014 = df[df['year'] == 2014]
print("Number of rows for year 2014:", len(rows_2014))
rows_2015 = df[df['year'] == 2015]
print("Number of rows for year 2015:", len(rows_2015))
rows_2016 = df[df['year'] == 2016]
print("Number of rows for year 2016:", len(rows_2016))
rows_2017 = df[df['year'] == 2017]
print("Number of rows for year 2017:", len(rows_2017))
rows_2018 = df[df['year'] == 2018]
print("Number of rows for year 2018:", len(rows_2018))
rows_2019 = df[df['year'] == 2019]
print("Number of rows for year 2019:", len(rows_2019))
rows_2020 = df[df['year'] == 2020]
print("Number of rows for year 2020:", len(rows_2020))
rows_2021 = df[df['year'] == 2021]
print("Number of rows for year 2021:", len(rows_2021))

other_years = df['year'][df['year'] != 2021].unique()
print("Other years present in the DataFrame:", other_years)

input_features = ['median_list_price', 'median_ppsf', 'avg_sale_to_list', 'inventory', 'median_dom', 'year', 'month','pending_sales', 'new_listings','off_market_in_two_weeks']
price_target = 'median_sale_price'
demand_target = 'homes_sold'

train_data = df[df['year'] < 2021]
test_data = df[df['year'] == 2021]

# Scaling features
scaler = StandardScaler()
X_train_price = scaler.fit_transform(train_data[input_features])
X_test_price = scaler.transform(test_data[input_features])

y_train_price = train_data[price_target]
y_test_price = test_data[price_target]
median_homes_sold = train_data[demand_target].median()
train_data.loc[:, 'high_demand'] = (train_data[demand_target] > median_homes_sold).astype(int)
test_data.loc[:, 'high_demand'] = (test_data[demand_target] > median_homes_sold).astype(int)
y_train_demand = train_data['high_demand']
y_test_demand = test_data['high_demand']

regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

regression_params = {
    'Linear Regression': {'fit_intercept': [True, False]},
    'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
    'Random Forest Regressor': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'Gradient Boosting Regressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

classification_models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
}

classification_params = {
    'Logistic Regression': {'penalty': ['l2'], 'C': [0.1, 1, 10]},
    'Random Forest Classifier': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'Gradient Boosting Classifier': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

def tune_and_evaluate(models, params, X_train, y_train, X_test, y_test, is_regression=True):
    best_models = {}
    results = pd.DataFrame(columns=['Model', 'Best Params', 'RMSE', 'MAE', 'MAPE', 'Accuracy', 'F1'])
    for name, model in models.items():
        grid_search = GridSearchCV(model, params[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        dump(best_model, f'best_{name.lower().replace(" ", "_")}_model.joblib')  # Save the best model
        check_is_fitted(best_model)
        predictions = best_model.predict(X_test)
        if is_regression:
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            new_row = pd.DataFrame({
            'Model': [name],
            'Best Params': [grid_search.best_params_],
            'RMSE': [rmse],
            'MAE': [mae],
            'MAPE': [mape] if is_regression else [np.nan],
            'Accuracy': [np.nan if is_regression else accuracy],
            'F1': [np.nan if is_regression else f1]
            })
            results = pd.concat([results, new_row], ignore_index=True)
        else:
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            new_row = pd.DataFrame({
            'Model': [name],
            'Best Params': [grid_search.best_params_],
            'RMSE': [rmse],
            'MAE': [mae],
            'MAPE': [mape] if is_regression else [np.nan],
            'Accuracy': [np.nan if is_regression else accuracy],
            'F1': [np.nan if is_regression else f1]
            })
            results = pd.concat([results, new_row], ignore_index=True)
    return best_models, results

best_models_regression, regression_results = tune_and_evaluate(regression_models, regression_params, X_train_price, y_train_price, X_test_price, y_test_price)

best_regression_model = regression_results.loc[regression_results['RMSE'].idxmin()]['Model']
dump(best_regression_model, 'best_regression_model.joblib')

print("Regression Results:")
print(regression_results[['Model', 'RMSE', 'MAE', 'MAPE']])

best_models_classification, classification_results = tune_and_evaluate(classification_models, classification_params, X_train_price, y_train_demand, X_test_price, y_test_demand, is_regression=False)

best_classification_model = classification_results.loc[classification_results['Accuracy'].idxmax()]['Model']
dump(best_classification_model, 'best_classification_model.joblib')

print("Classification Results:")
print(classification_results[['Model', 'Accuracy', 'F1']])

best_classification_model = best_models_classification['Random Forest Classifier']  # replace with the actual best model's name if different
# test_data['probability_high_demand'] = best_classification_model.predict_proba(X_test_price)[:, 1]
# top_regions_demand = test_data[['region', 'probability_high_demand']].groupby('region').mean().sort_values(by='probability_high_demand', ascending=False).head(10)
# print("Top 10 regions in demand:")
# print(top_regions_demand)
# Make sure to use .loc to assign values to the DataFrame to avoid SettingWithCopyWarning
test_data.loc[:, 'probability_high_demand'] = best_classification_model.predict_proba(X_test_price)[:, 1]

region_demand = test_data.groupby('region')['probability_high_demand'].mean().reset_index()

top_regions = region_demand.sort_values(by='probability_high_demand', ascending=False).head(10)

# Print the top 10 regions in demand
print("Top 10 regions in demand:")
print(top_regions)

