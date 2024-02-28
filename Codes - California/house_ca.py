from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

housing_data = pd.read_csv("/Users/YouShallNotPass/Desktop/BD with ML Project/Model-Train/CA/filtered_dataset_CA.csv")

print(housing_data.info())
print(housing_data.head())

columns_to_drop = ['region_type', 'region_type_id', 'table_id', 'state', 'state_code', 'last_updated']
housing_data_cleaned = housing_data.drop(columns=columns_to_drop)

housing_data_cleaned = housing_data_cleaned.dropna()

housing_data['period_begin'] = pd.to_datetime(housing_data['period_begin'])
housing_data['year'] = housing_data['period_begin'].dt.year
housing_data['month'] = housing_data['period_begin'].dt.month

# housing_data.drop(['period_begin'], axis=1)

num_rows = housing_data.shape[0]
print("Number of rows after removing years:", num_rows)

rows_2012 = housing_data[housing_data['year'] == 2012]
print("Number of rows for year 2012:", len(rows_2012))
rows_2013 = housing_data[housing_data['year'] == 2013]
print("Number of rows for year 2013:", len(rows_2013))
rows_2014 = housing_data[housing_data['year'] == 2014]
print("Number of rows for year 2014:", len(rows_2014))
rows_2015 = housing_data[housing_data['year'] == 2015]
print("Number of rows for year 2015:", len(rows_2015))
rows_2016 = housing_data[housing_data['year'] == 2016]
print("Number of rows for year 2016:", len(rows_2016))
rows_2017 = housing_data[housing_data['year'] == 2017]
print("Number of rows for year 2017:", len(rows_2017))
rows_2018 = housing_data[housing_data['year'] == 2018]
print("Number of rows for year 2018:", len(rows_2018))
rows_2019 = housing_data[housing_data['year'] == 2019]
print("Number of rows for year 2019:", len(rows_2019))
rows_2020 = housing_data[housing_data['year'] == 2020]
print("Number of rows for year 2020:", len(rows_2020))
rows_2021 = housing_data[housing_data['year'] == 2021]
print("Number of rows for year 2021:", len(rows_2021))

other_years = housing_data['year'][housing_data['year'] != 2021].unique()
print("Other years present in the DataFrame:", other_years)

print(housing_data.head())

print(housing_data.describe())

print(housing_data.dtypes)

# Check for missing values
print(housing_data.isnull().sum())

# Histogram for 'median_sale_price'
plt.hist(housing_data['median_sale_price'], bins=30)
plt.title('Distribution of Median Sale Price')
plt.xlabel('Median Sale Price')
plt.ylabel('Frequency')
plt.show()


correlation_matrix = housing_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

plt.scatter(housing_data['median_sale_price'], housing_data['homes_sold'])
plt.title('Median Sale Price vs Homes Sold')
plt.xlabel('Median Sale Price')
plt.ylabel('Homes Sold')
plt.show()

# Box plot for 'median_sale_price'
sns.boxplot(housing_data['median_sale_price'])
plt.title('Box Plot for Median Sale Price')
plt.show()

housing_data_cleaned['market_hotness'] = housing_data_cleaned['median_sale_price'] / housing_data_cleaned['median_list_price']

numerical_cols = housing_data_cleaned.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = housing_data_cleaned.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

housing_data_transformed = preprocessor.fit_transform(housing_data_cleaned)


ohe_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

transformed_columns = list(numerical_cols) + list(ohe_columns)

# print(transformed_columns)


housing_data_dense = housing_data_transformed.toarray()

X = pd.DataFrame(housing_data_dense, columns=transformed_columns)

X_sparse = housing_data_transformed

# Split the data
y_price = housing_data_cleaned['median_sale_price']
y_demand = housing_data_cleaned['homes_sold']


X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_sparse, y_price, test_size=0.2, random_state=42)
X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(X_sparse, y_demand, test_size=0.2, random_state=0)

model_price = LinearRegression()
model_price.fit(X_train_price, y_train_price)

y_pred_price = model_price.predict(X_test_price)
rmse_price = mean_squared_error(y_test_price, y_pred_price, squared=False)
mse_price = mean_squared_error(y_test_price, y_pred_price)
mae_price = mean_absolute_error(y_test_price, y_pred_price)
r2_price = r2_score(y_test_price, y_pred_price)

model_demand = LinearRegression()
model_demand.fit(X_train_demand, y_train_demand)

y_pred_demand = model_demand.predict(X_test_demand)
rmse_demand = mean_squared_error(y_test_demand, y_pred_demand, squared=False)
mse_demand = mean_squared_error(y_test_demand, y_pred_demand)
mae_demand = mean_absolute_error(y_test_demand, y_pred_demand)
r2_demand = r2_score(y_test_demand, y_pred_demand)

print(f'RMSE for Price Prediction: {rmse_price}')
print(f'MSE for Price Prediction: {mse_price}')
print(f'MAE for Price Prediction: {mae_price}')
print(f'R2 for Price Prediction: {r2_price}')

print(f'RMSE for Demand Prediction: {rmse_price}')
print(f'MSE for Demand Prediction: {mse_price}')
print(f'MAE for Demand Prediction: {mae_price}')
print(f'R2 for Demand Prediction: {r2_price}')

most_demand = housing_data_cleaned.groupby('region')['homes_sold'].sum().sort_values(ascending=False)

print(most_demand.head())  

def train_evaluate_save_model(model, params, X_train, X_test, y_train, y_test, model_name):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    train_predictions = best_model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)

    test_predictions = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Training - MSE: {train_mse}, RMSE: {train_rmse}, MAE: {train_mae}, R2: {train_r2}")
    print(f"Test - MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, R2: {test_r2}")

    joblib.dump(best_model, f'{model_name}.pkl')
    print(f"Model saved as {model_name}.pkl")
    return best_model

params_dt_price = {'max_depth': [20], 'min_samples_split': [5]}

model_dt_price = train_evaluate_save_model(
    DecisionTreeRegressor(),
    params_dt_price,
    X_train_price,
    X_test_price,
    y_train_price,
    y_test_price,
    "Decision_Tree_price_model"
)

params_rf_price = {'n_estimators': [100], 'max_depth': [10]}
model_rf_price = train_evaluate_save_model(RandomForestRegressor(), params_rf_price, X_train_price, X_test_price, y_train_price, y_test_price,"Random_Forest_price_model")

params_xgb_price = {'n_estimators': [100], 'learning_rate': [0.1]}
model_xgb_price = train_evaluate_save_model(XGBRegressor(), params_xgb_price, X_train_price, X_test_price, y_train_price, y_test_price,"XGBoost_price_model")

params_dt_demand = {'max_depth': [20], 'min_samples_split': [10]}

model_dt_demand = train_evaluate_save_model(
    DecisionTreeRegressor(),
    params_dt_demand,
    X_train_demand,
    X_test_demand,
    y_train_demand,
    y_test_demand,
    "Decision_Tree_demand_model"
)

params_rf_demand = {'n_estimators': [100], 'max_depth': [20]}
model_rf_demand = train_evaluate_save_model(RandomForestRegressor(), params_rf_demand, X_train_demand, X_test_demand, y_train_demand, y_test_demand,"Random_Forest_demand_model")

params_xgb_demand = {'n_estimators': [100], 'learning_rate': [0.1]}
model_xgb_demand = train_evaluate_save_model(XGBRegressor(), params_xgb_demand, X_train_demand, X_test_demand, y_train_demand, y_test_demand,"XGBoost_demand_model")