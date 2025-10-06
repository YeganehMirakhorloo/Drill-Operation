from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_baseline_models(data):
    """Train baseline models for comparison"""
    
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # ROP prediction
    print("\n=== ROP Baseline Models ===")
    y_rop = data['ROP'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_rop, test_size=0.2, random_state=42
    )
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, y_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
    
    results['rop'] = {'ridge_r2': ridge_r2, 'rf_r2': rf_r2}
    
    # Torque prediction
    print("\n=== Torque Baseline Models ===")
    y_torque = data['Surface_Torque'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_torque, test_size=0.2, random_state=42
    )
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, y_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
    
    results['torque'] = {'ridge_r2': ridge_r2, 'rf_r2': rf_r2}
    
    return results
