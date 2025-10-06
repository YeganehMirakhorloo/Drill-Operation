from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_baseline_models(data):
    """Train baseline models for comparison - works with available columns"""

    # Define all possible features and use only what's available
    all_possible_features = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load', 'Vis', 'Mw', 'Mob', 'Gpm']
    feature_columns = [col for col in all_possible_features if col in data.columns]
    
    if len(feature_columns) < 3:
        raise ValueError(f"Need at least 3 features, found only {len(feature_columns)}: {feature_columns}")
    
    print(f"\n📊 Using {len(feature_columns)} features: {feature_columns}")
    
    X = data[feature_columns].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ==================== ROP PREDICTION ====================
    if 'ROP' in data.columns:
        print("\n" + "="*70)
        print("ROP BASELINE MODELS")
        print("="*70)
        
        y_rop = data['ROP'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_rop, test_size=0.2, random_state=42
        )

        # Ridge Regression for ROP
        print("\n[1] Ridge Regression (ROP):")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        ridge_r2 = r2_score(y_test, y_pred)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    R² Score: {ridge_r2:.4f}")
        print(f"    RMSE:     {ridge_rmse:.4f}")

        # Random Forest for ROP
        print("\n[2] Random Forest (ROP):")
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, y_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    R² Score: {rf_r2:.4f}")
        print(f"    RMSE:     {rf_rmse:.4f}")

        results['rop'] = {
            'ridge_r2': ridge_r2, 
            'ridge_rmse': ridge_rmse,
            'rf_r2': rf_r2,
            'rf_rmse': rf_rmse
        }

    # ==================== TORQUE PREDICTION ====================
    if 'Surface_Torque' in data.columns:
        print("\n" + "="*70)
        print("TORQUE BASELINE MODELS")
        print("="*70)
        
        y_torque = data['Surface_Torque'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_torque, test_size=0.2, random_state=42
        )

        # Ridge Regression for Torque
        print("\n[1] Ridge Regression (Torque):")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        ridge_r2 = r2_score(y_test, y_pred)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    R² Score: {ridge_r2:.4f}")
        print(f"    RMSE:     {ridge_rmse:.4f}")

        # Random Forest for Torque
        print("\n[2] Random Forest (Torque):")
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, y_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    R² Score: {rf_r2:.4f}")
        print(f"    RMSE:     {rf_rmse:.4f}")

        results['torque'] = {
            'ridge_r2': ridge_r2,
            'ridge_rmse': ridge_rmse,
            'rf_r2': rf_r2,
            'rf_rmse': rf_rmse
        }

    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("BASELINE MODELS SUMMARY")
    print("="*70)
    
    if 'rop' in results:
        print(f"\nROP Models:")
        print(f"  Ridge:         R²={results['rop']['ridge_r2']:.4f}, RMSE={results['rop']['ridge_rmse']:.4f}")
        print(f"  Random Forest: R²={results['rop']['rf_r2']:.4f}, RMSE={results['rop']['rf_rmse']:.4f}")
    
    if 'torque' in results:
        print(f"\nTorque Models:")
        print(f"  Ridge:         R²={results['torque']['ridge_r2']:.4f}, RMSE={results['torque']['ridge_rmse']:.4f}")
        print(f"  Random Forest: R²={results['torque']['rf_r2']:.4f}, RMSE={results['torque']['rf_rmse']:.4f}")
    
    print("="*70 + "\n")

    return results
