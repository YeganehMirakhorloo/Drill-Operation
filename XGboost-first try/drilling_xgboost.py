# drilling_xgboost.py
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class DrillingXGBoostPredictor:
    """XGBoost-based drilling parameter predictor"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, gamma=0, 
                 min_child_weight=1, reg_alpha=0, reg_lambda=1):
        """
        Initialize XGBoost predictor with hyperparameters
        
        Parameters:
        -----------
        n_estimators : int, number of boosting rounds
        max_depth : int, maximum tree depth
        learning_rate : float, step size shrinkage
        subsample : float, subsample ratio of training instances
        colsample_bytree : float, subsample ratio of columns
        gamma : float, minimum loss reduction for split
        min_child_weight : int, minimum sum of instance weight in child
        reg_alpha : float, L1 regularization term
        reg_lambda : float, L2 regularization term
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.best_iteration = None
    
    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the XGBoost model with validation set
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        print(f"Starting training with LR={self.learning_rate}, Max Depth={self.max_depth}, N_estimators={self.n_estimators}...")
        
        # Convert to numpy arrays if needed
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Scale the data
        X_train_scaled, y_train_scaled = self.prepare_data(X_train, y_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1
        )
        
        # Train WITHOUT early stopping
        self.model.fit(
            X_train_scaled, 
            y_train_scaled,
            eval_set=[(X_val_scaled, y_val_scaled)],
            verbose=False
        )
        
        print("✓ Training completed!")

    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate AARE (Average Absolute Relative Error)
        mask = y_test != 0
        if np.any(mask):
            aare = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
        else:
            aare = np.inf
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'aare': aare,
            'predictions': predictions
        }
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.feature_importances_


def train_drilling_models(data):
    """Train both ROP and Torque models using XGBoost"""
    
    # Prepare features and targets
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    
    # Train ROP model
    print("=" * 60)
    print("Training ROP Model with XGBoost...")
    print("=" * 60)
    y_rop = data['ROP'].values
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )
    
    rop_model = DrillingXGBoostPredictor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        min_child_weight=1,
        reg_alpha=0,
        reg_lambda=1
    )
    
    # Split training data for validation
    X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
        X_train, y_rop_train, test_size=0.2, random_state=42
    )
    
    rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
    rop_results = rop_model.evaluate(X_test, y_rop_test)
    
    print(f"\nROP Model Performance:")
    print(f"  RMSE: {rop_results['rmse']:.4f}")
    print(f"  R²: {rop_results['r2']:.4f}")
    print(f"  AARE: {rop_results['aare']:.2f}%")
    
    # Train Torque model
    print("\n" + "=" * 60)
    print("Training Torque Model with XGBoost...")
    print("=" * 60)
    y_torque = data['Surface_Torque'].values
    X_train, X_test, y_torque_train, y_torque_test = train_test_split(
        X, y_torque, test_size=0.2, random_state=42
    )
    
    torque_model = DrillingXGBoostPredictor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        min_child_weight=1,
        reg_alpha=0,
        reg_lambda=1
    )
    
    X_train_split, X_val_split, y_torque_train_split, y_torque_val_split = train_test_split(
        X_train, y_torque_train, test_size=0.2, random_state=42
    )
    
    torque_model.train(X_train_split, y_torque_train_split, X_val_split, y_torque_val_split)
    torque_results = torque_model.evaluate(X_test, y_torque_test)
    
    print(f"\nTorque Model Performance:")
    print(f"  RMSE: {torque_results['rmse']:.4f}")
    print(f"  R²: {torque_results['r2']:.4f}")
    print(f"  AARE: {torque_results['aare']:.2f}%")
    
    return rop_model, torque_model, rop_results, torque_results
