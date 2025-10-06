# drilling_random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class DrillingRandomForestPredictor:
    """Random Forest-based drilling parameter predictor"""

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt', bootstrap=True,
                 oob_score=True, random_state=42, n_jobs=-1):
        """
        Initialize Random Forest predictor with hyperparameters

        Parameters:
        -----------
        n_estimators : int, number of trees in the forest
        max_depth : int, maximum tree depth
        min_samples_split : int, minimum samples to split a node
        min_samples_leaf : int, minimum samples in leaf node
        max_features : str or int, number of features to consider at each split
        bootstrap : bool, whether to use bootstrap samples
        oob_score : bool, whether to use out-of-bag samples to estimate R²
        random_state : int, random seed
        n_jobs : int, number of parallel jobs
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the Random Forest model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (kept for API compatibility, not used in RF)
            y_val: Validation target (kept for API compatibility, not used in RF)
        """
        print(f"Starting training with n_estimators={self.n_estimators}, max_depth={self.max_depth}...")

        # Convert to numpy arrays if needed
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale the data
        X_train_scaled, y_train_scaled = self.prepare_data(X_train, y_train)

        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )

        # Train
        self.model.fit(X_train_scaled, y_train_scaled)

        # Print OOB score if available
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            print(f"✓ Out-of-Bag R² Score: {self.model.oob_score_:.4f}")

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
    """Train both ROP and Torque models using Random Forest"""

    # Prepare features and targets
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values

    # Train ROP model
    print("=" * 60)
    print("Training ROP Model with Random Forest...")
    print("=" * 60)
    y_rop = data['ROP'].values
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )

    rop_model = DrillingRandomForestPredictor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True
    )

    # Split training data for validation (API compatibility)
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
    print("Training Torque Model with Random Forest...")
    print("=" * 60)
    y_torque = data['Surface_Torque'].values
    X_train, X_test, y_torque_train, y_torque_test = train_test_split(
        X, y_torque, test_size=0.2, random_state=42
    )

    torque_model = DrillingRandomForestPredictor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True
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
