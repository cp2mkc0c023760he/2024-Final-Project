from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_data, preprocess_data
import xgboost as xgb
import numpy as np

def find_best_hyperparams(ticker, file_path):
    # Load and preprocess data
    dataset = load_data(file_path)
    train_data, validation_data = preprocess_data(dataset)

    # Creating features and target
    X_train = train_data.drop(columns=[ticker]).values
    y_train = train_data[ticker].values

 # Define XGBoost parameters
    params = {
        'objective': ['reg:squarederror'],
        'eval_metric': ['rmse'],
        'max_depth': [3, 5, 7],
        'eta': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'lambda': [0.1, 0.3, 1.0],
        'alpha': [0.1, 0.3, 1.0],
        'seed': [42]  
    }

    # Create XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')  # Keeping objective fixed


    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=params,
                               scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)
    print("Best Parameters:", best_params)
    print("Best RMSE:", best_score)

find_best_hyperparams('EURUSD', 'Data/Forex-preprocessed/currencies.csv')

#{'colsample_bytree': 0.8, 'eta': 0.1, 'eval_metric': 'rmse', 'max_depth': 5, 'seed': 42, 'subsample': 0.6}
#{'alpha': 0.1, 'colsample_bytree': 0.8, 'eta': 0.1, 'eval_metric': 'rmse', 'gamma': 0, 'lambda': 0.3, 'max_depth': 3, 'objective': 'reg:squarederror', 'seed': 42, 'subsample': 1.0}