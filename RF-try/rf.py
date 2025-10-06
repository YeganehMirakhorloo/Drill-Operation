import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DrillingOptimizationRF:
    """
    Drilling optimization using Random Forest
    Adapted from: Processes 2025, 13, 1472 (XGBoost version)
    """

    def __init__(self):
        self.rop_model = None
        self.torque_model = None
        self.scaler_rop = StandardScaler()
        self.scaler_torque = StandardScaler()

    def load_data(self, train_path, test_path, train_header_row=2, test_header_row=0):
        """
        Load training and testing data from CSV or Excel files
        """
        print("\n" + "="*70)
        print("Loading data...")
        print("="*70)

        # Load training data
        if train_path.endswith('.csv'):
            self.train_data = pd.read_csv(train_path, header=train_header_row)
        else:
            self.train_data = pd.read_excel(train_path, header=train_header_row)

        # Load testing data
        if test_path.endswith('.csv'):
            self.test_data = pd.read_csv(test_path, header=test_header_row)
        else:
            self.test_data = pd.read_excel(test_path, header=test_header_row)

        # Clean column names
        self.train_data.columns = self.train_data.columns.str.strip()
        self.test_data.columns = self.test_data.columns.str.strip()

        print(f"✓ Training samples: {len(self.train_data)}")
        print(f"✓ Testing samples: {len(self.test_data)}")
        print(f"\nTraining columns: {list(self.train_data.columns)}")
        print(f"Testing columns: {list(self.test_data.columns)}")

        print(f"\nFirst few rows of training data:")
        print(self.train_data.head())

        return self.train_data, self.test_data

    def clean_column_names(self, df):
        """
        Standardize column names
        """
        column_mapping = {
            'Rop': 'ROP', 'rop': 'ROP',
            'Wob': 'WOB', 'wob': 'WOB',
            'Rpm': 'RPM', 'rpm': 'RPM',
            'Gpm': 'GPM', 'gpm': 'GPM',
            'Spp': 'SPP', 'spp': 'SPP',
            'Mw': 'MW', 'mw': 'MW',
            'Vis': 'Vis', 'vis': 'Vis',
            'Mob': 'Mob', 'mob': 'Mob',
            'depth': 'Depth', 'DEPTH': 'Depth',
            'torque': 'Torque', 'TORQUE': 'Torque'
        }

        df = df.rename(columns=column_mapping)
        return df

    def _add_interaction_features(self):
        """
        Add interaction features to improve model performance
        """
        # For ROP model
        if 'WOB' in self.X_train_rop.columns and 'RPM' in self.X_train_rop.columns:
            self.X_train_rop['WOB_RPM_ratio'] = self.X_train_rop['WOB'] / (self.X_train_rop['RPM'] + 1e-6)
            self.X_test_rop['WOB_RPM_ratio'] = self.X_test_rop['WOB'] / (self.X_test_rop['RPM'] + 1e-6)
            
        if 'GPM' in self.X_train_rop.columns and 'RPM' in self.X_train_rop.columns:
            self.X_train_rop['GPM_RPM_ratio'] = self.X_train_rop['GPM'] / (self.X_train_rop['RPM'] + 1e-6)
            self.X_test_rop['GPM_RPM_ratio'] = self.X_test_rop['GPM'] / (self.X_test_rop['RPM'] + 1e-6)
            
        if 'WOB' in self.X_train_rop.columns and 'RPM' in self.X_train_rop.columns:
            self.X_train_rop['WOB_RPM_interaction'] = self.X_train_rop['WOB'] * self.X_train_rop['RPM']
            self.X_test_rop['WOB_RPM_interaction'] = self.X_test_rop['WOB'] * self.X_test_rop['RPM']

    def _remove_outliers(self, df, columns, factor=1.5):
        """
        Remove outliers using IQR method
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean

    def preprocess_data(self, rop_input_features, rop_output,
                       torque_input_features=None, torque_output=None,
                       remove_outliers=True, outlier_factor=1.5):
        """
        Preprocess data for both models
        """
        print("\n" + "="*70)
        print("Preprocessing data...")
        print("="*70)

        # Clean column names
        train_clean = self.clean_column_names(self.train_data.copy())
        test_clean = self.clean_column_names(self.test_data.copy())

        # Remove non-numeric columns
        cols_to_drop = ['Date', 'Time', 'HOB', 'Lit', 'Vis']
        for col in cols_to_drop:
            if col in train_clean.columns:
                train_clean = train_clean.drop(columns=[col])
            if col in test_clean.columns:
                test_clean = test_clean.drop(columns=[col])

        # Convert to numeric
        for col in train_clean.columns:
            train_clean[col] = pd.to_numeric(train_clean[col], errors='coerce')
            if col in test_clean.columns:
                test_clean[col] = pd.to_numeric(test_clean[col], errors='coerce')

        # Remove nulls
        original_train_size = len(train_clean)
        original_test_size = len(test_clean)

        train_clean = train_clean.dropna()
        test_clean = test_clean.dropna()

        print(f"\n✓ Training samples after null removal: {len(train_clean)} (removed {original_train_size - len(train_clean)})")
        print(f"✓ Testing samples after null removal: {len(test_clean)} (removed {original_test_size - len(test_clean)})")

        # Remove outliers only from training data
        if remove_outliers:
            print(f"\nRemoving outliers with IQR factor = {outlier_factor}...")
            original_size = len(train_clean)

            all_features = list(set(rop_input_features + [rop_output]))
            if torque_input_features:
                all_features.extend(torque_input_features + [torque_output])
            all_features = list(set(all_features))

            all_features = [f for f in all_features if f in train_clean.columns]

            train_clean = self._remove_outliers(train_clean, all_features, outlier_factor)

            removed = original_size - len(train_clean)
            print(f"✓ Removed {removed} outliers ({removed/original_size*100:.2f}%)")

        # Prepare data for ROP model
        print(f"\nPreparing ROP model data...")

        missing_cols = [col for col in rop_input_features if col not in train_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in training data: {missing_cols}")

        self.X_train_rop = train_clean[rop_input_features].copy()
        self.y_train_rop = train_clean[rop_output].copy()
        self.X_test_rop = test_clean[rop_input_features].copy()
        self.y_test_rop = test_clean[rop_output].copy()

        # Add interaction features
        self._add_interaction_features()

        # Normalize
        self.X_train_rop_scaled = self.scaler_rop.fit_transform(self.X_train_rop)
        self.X_test_rop_scaled = self.scaler_rop.transform(self.X_test_rop)

        print(f"  ✓ Training shape: {self.X_train_rop_scaled.shape}")
        print(f"  ✓ Testing shape: {self.X_test_rop_scaled.shape}")

        # Prepare data for Torque model
        if torque_input_features:
            print(f"\nPreparing Torque model data...")

            missing_cols = [col for col in torque_input_features if col not in train_clean.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in training data: {missing_cols}")

            self.X_train_torque = train_clean[torque_input_features].copy()
            self.y_train_torque = train_clean[torque_output].copy()
            self.X_test_torque = test_clean[torque_input_features].copy()
            self.y_test_torque = test_clean[torque_output].copy()

            # Normalize
            self.X_train_torque_scaled = self.scaler_torque.fit_transform(self.X_train_torque)
            self.X_test_torque_scaled = self.scaler_torque.transform(self.X_test_torque)

            print(f"  ✓ Training shape: {self.X_train_torque_scaled.shape}")
            print(f"  ✓ Testing shape: {self.X_test_torque_scaled.shape}")

    def check_data_quality(self):
        """
        Compare training and testing data distributions
        """
        print("\n" + "="*70)
        print("DATA QUALITY CHECK")
        print("="*70)
        
        print("\nROP Features - Mean Comparison:")
        print(f"{'Feature':<15} {'Train Mean':>12} {'Test Mean':>12} {'Difference':>12}")
        print("-" * 55)
        
        for col in self.X_train_rop.columns:
            train_mean = self.X_train_rop[col].mean()
            test_mean = self.X_test_rop[col].mean()
            diff = abs(train_mean - test_mean) / train_mean * 100 if train_mean != 0 else 0
            print(f"{col:<15} {train_mean:>12.2f} {test_mean:>12.2f} {diff:>11.1f}%")
        
        print(f"\n{'ROP (target)':<15} {self.y_train_rop.mean():>12.2f} {self.y_test_rop.mean():>12.2f} {abs(self.y_train_rop.mean() - self.y_test_rop.mean())/self.y_train_rop.mean()*100:>11.1f}%")
        print("="*70)

    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        aare = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        return {'R2': r2, 'RMSE': rmse, 'AARE': aare}

    def train_rop_model_with_cv(self):
        """
        Train ROP model with cross-validation to check generalization
        """
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }

        model = RandomForestRegressor(**params)

        # 5-fold cross-validation on training data
        cv_scores = cross_val_score(
            model,
            self.X_train_rop_scaled,
            self.y_train_rop,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )

        print("\n" + "="*70)
        print("Cross-Validation Results (on Training Data)")
        print("="*70)
        print(f"CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        if cv_scores.mean() > 0.6:
            print("✓ Model can generalize well on training data")
            print("⚠️ If test R² is still bad, your test set is OUT-OF-DISTRIBUTION!")
        else:
            print("⚠️ Model struggles even with cross-validation")

        print("="*70)

        return cv_scores

    def train_rop_model(self, params=None, verbose=True):
        """Train Random Forest model for ROP prediction with conservative parameters"""
        if params is None:
            params = {
                'n_estimators': 200,           # Number of trees
                'max_depth': 15,               # Maximum depth of trees
                'min_samples_split': 10,       # Minimum samples to split a node
                'min_samples_leaf': 5,         # Minimum samples in leaf node
                'max_features': 'sqrt',        # Number of features to consider at each split
                'bootstrap': True,             # Use bootstrap samples
                'oob_score': True,             # Use out-of-bag samples to estimate R²
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            }

        print("\n" + "="*70)
        print("Training ROP model with Random Forest (Conservative)...")
        print("="*70)

        self.rop_model = RandomForestRegressor(**params)
        self.rop_model.fit(self.X_train_rop_scaled, self.y_train_rop)

        if hasattr(self.rop_model, 'oob_score_'):
            print(f"\n✓ Out-of-Bag R² Score: {self.rop_model.oob_score_:.4f}")

        # Predictions
        y_train_pred_rop = self.rop_model.predict(self.X_train_rop_scaled)
        y_test_pred_rop = self.rop_model.predict(self.X_test_rop_scaled)

        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train_rop, y_train_pred_rop)
        test_metrics = self._calculate_metrics(self.y_test_rop, y_test_pred_rop)

        self.rop_results = {
            'train': train_metrics,
            'test': test_metrics,
            'y_train_pred': y_train_pred_rop,
            'y_test_pred': y_test_pred_rop
        }

        print(f"\nROP Model Results:")
        print(f"  {'Dataset':<10} {'R²':>8} {'RMSE':>10} {'AARE':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Train':<10} {train_metrics['R2']:>8.4f} {train_metrics['RMSE']:>10.4f} {train_metrics['AARE']:>9.2f}%")
        print(f"  {'Test':<10} {test_metrics['R2']:>8.4f} {test_metrics['RMSE']:>10.4f} {test_metrics['AARE']:>9.2f}%")

        return self.rop_model, self.rop_results

    def train_torque_model(self, params=None, verbose=True):
        """Train Random Forest model for Torque prediction with conservative parameters"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 15,       # Slightly more conservative
                'min_samples_leaf': 8,         # Slightly more conservative
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            }

        print("\n" + "="*70)
        print("Training Torque model with Random Forest (Conservative)...")
        print("="*70)

        self.torque_model = RandomForestRegressor(**params)
        self.torque_model.fit(self.X_train_torque_scaled, self.y_train_torque)

        if hasattr(self.torque_model, 'oob_score_'):
            print(f"\n✓ Out-of-Bag R² Score: {self.torque_model.oob_score_:.4f}")

        # Predictions
        y_train_pred_torque = self.torque_model.predict(self.X_train_torque_scaled)
        y_test_pred_torque = self.torque_model.predict(self.X_test_torque_scaled)

        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train_torque, y_train_pred_torque)
        test_metrics = self._calculate_metrics(self.y_test_torque, y_test_pred_torque)

        self.torque_results = {
            'train': train_metrics,
            'test': test_metrics,
            'y_train_pred': y_train_pred_torque,
            'y_test_pred': y_test_pred_torque
        }

        print(f"\nTorque Model Results:")
        print(f"  {'Dataset':<10} {'R²':>8} {'RMSE':>10} {'AARE':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Train':<10} {train_metrics['R2']:>8.4f} {train_metrics['RMSE']:>10.4f} {train_metrics['AARE']:>9.2f}%")
        print(f"  {'Test':<10} {test_metrics['R2']:>8.4f} {test_metrics['RMSE']:>10.4f} {test_metrics['AARE']:>9.2f}%")

        return self.torque_model, self.torque_results

    def plot_results(self, save_path=None, dpi=300):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ROP Training
        axes[0, 0].scatter(self.y_train_rop, self.rop_results['y_train_pred'], 
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[0, 0].plot([self.y_train_rop.min(), self.y_train_rop.max()],
                       [self.y_train_rop.min(), self.y_train_rop.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual ROP', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted ROP', fontsize=12, fontweight='bold')
        axes[0, 0].set_title(f'ROP - Training Set (R² = {self.rop_results["train"]["R2"]:.4f})',
                            fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # ROP Testing
        axes[0, 1].scatter(self.y_test_rop, self.rop_results['y_test_pred'],
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5, color='orange')
        axes[0, 1].plot([self.y_test_rop.min(), self.y_test_rop.max()],
                       [self.y_test_rop.min(), self.y_test_rop.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual ROP', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Predicted ROP', fontsize=12, fontweight='bold')
        axes[0, 1].set_title(f'ROP - Testing Set (R² = {self.rop_results["test"]["R2"]:.4f})',
                            fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Torque Training
        axes[1, 0].scatter(self.y_train_torque, self.torque_results['y_train_pred'],
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5, color='green')
        axes[1, 0].plot([self.y_train_torque.min(), self.y_train_torque.max()],
                       [self.y_train_torque.min(), self.y_train_torque.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Torque', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Predicted Torque', fontsize=12, fontweight='bold')
        axes[1, 0].set_title(f'Torque - Training Set (R² = {self.torque_results["train"]["R2"]:.4f})',
                            fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Torque Testing
        axes[1, 1].scatter(self.y_test_torque, self.torque_results['y_test_pred'],
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.5, color='purple')
        axes[1, 1].plot([self.y_test_torque.min(), self.y_test_torque.max()],
                       [self.y_test_torque.min(), self.y_test_torque.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Torque', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Predicted Torque', fontsize=12, fontweight='bold')
        axes[1, 1].set_title(f'Torque - Testing Set (R² = {self.torque_results["test"]["R2"]:.4f})',
                            fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nPlot saved: {save_path}")

        plt.show()

    def plot_feature_importance(self, save_path=None, dpi=300):
        """Plot feature importance"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ROP Model
        importance_rop = self.rop_model.feature_importances_
        features_rop = self.X_train_rop.columns
        indices_rop = np.argsort(importance_rop)[::-1]

        axes[0].barh(range(len(indices_rop)), importance_rop[indices_rop], color='skyblue')
        axes[0].set_yticks(range(len(indices_rop)))
        axes[0].set_yticklabels([features_rop[i] for i in indices_rop])
        axes[0].set_xlabel('Importance (Mean Decrease in Impurity)', fontsize=12, fontweight='bold')
        axes[0].set_title('ROP Model - Feature Importance', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Torque Model
        importance_torque = self.torque_model.feature_importances_
        features_torque = self.X_train_torque.columns
        indices_torque = np.argsort(importance_torque)[::-1]

        axes[1].barh(range(len(indices_torque)), importance_torque[indices_torque], color='lightcoral')
        axes[1].set_yticks(range(len(indices_torque)))
        axes[1].set_yticklabels([features_torque[i] for i in indices_torque])
        axes[1].set_xlabel('Importance (Mean Decrease in Impurity)', fontsize=12, fontweight='bold')
        axes[1].set_title('Torque Model - Feature Importance', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFeature importance plot saved: {save_path}")

        plt.show()

    def generate_report(self, save_path=None):
        """Generate comprehensive report"""
        report_data = {
            'Metric': ['RMSE', 'AARE (%)', 'R²'],
            'ROP (Train)': [
                self.rop_results['train']['RMSE'],
                self.rop_results['train']['AARE'],
                self.rop_results['train']['R2']
            ],
            'ROP (Test)': [
                self.rop_results['test']['RMSE'],
                self.rop_results['test']['AARE'],
                self.rop_results['test']['R2']
            ],
            'Torque (Train)': [
                self.torque_results['train']['RMSE'],
                self.torque_results['train']['AARE'],
                self.torque_results['train']['R2']
            ],
            'Torque (Test)': [
                self.torque_results['test']['RMSE'],
                self.torque_results['test']['AARE'],
                self.torque_results['test']['R2']
            ]
        }

        report_df = pd.DataFrame(report_data)

        print("\n" + "="*80)
        print("Comprehensive Report - Random Forest Results")
        print("="*80)
        print(report_df.to_string(index=False))
        print("="*80)

        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"\nReport saved: {save_path}")

        return report_df

    def save_models(self, rop_path='rop_model_rf.pkl', torque_path='torque_model_rf.pkl'):
        """Save trained models using pickle"""
        import pickle
        
        with open(rop_path, 'wb') as f:
            pickle.dump(self.rop_model, f)
        
        with open(torque_path, 'wb') as f:
            pickle.dump(self.torque_model, f)

        print(f"\nModels saved:")
        print(f"  - ROP model: {rop_path}")
        print(f"  - Torque model: {torque_path}")

# =============================================================================
# Example usage with your data files
# =============================================================================

if __name__ == "__main__":

    # Create optimizer object
    optimizer = DrillingOptimizationRF()

    # Load data
    train_data, test_data = optimizer.load_data(
        train_path=r'E:\Data\pure\drill operation\human edit\Bit Data#1214#RR#34.xlsx',
        test_path=r'E:\Data\pure\drill operation\human edit\Bit Data#1214#MI#131.xlsx',
        train_header_row=0,
        test_header_row=2
    )

    # Define features
    rop_features = ['Depth', 'Mob', 'RPM', 'WOB', 'Torque', 'GPM', 'SPP', 'MW']
    rop_target = 'ROP'

    torque_features = ['Depth', 'Mob', 'RPM', 'WOB', 'ROP', 'GPM', 'SPP', 'MW']
    torque_target = 'Torque'

    # Preprocess data
    optimizer.preprocess_data(
        rop_input_features=rop_features,
        rop_output=rop_target,
        torque_input_features=torque_features,
        torque_output=torque_target,
        remove_outliers=True,
        outlier_factor=1.5
    )

    # ⭐ CHECK DATA QUALITY FIRST
    optimizer.check_data_quality()

    # ⭐ RUN CROSS-VALIDATION TO SEE IF MODEL CAN LEARN
    optimizer.train_rop_model_with_cv()

    # Train models with Random Forest
    rop_model, rop_results = optimizer.train_rop_model(verbose=False)
    torque_model, torque_results = optimizer.train_torque_model(verbose=False)

    # Plot results
    optimizer.plot_results(save_path='results_comparison_rf.png', dpi=300)
    optimizer.plot_feature_importance(save_path='feature_importance_rf.png', dpi=300)

    # Generate report
    report = optimizer.generate_report(save_path='final_report_rf.csv')

    # Save models
    optimizer.save_models()

    print("\n" + "="*70)
    print("All steps completed successfully!")
    print("="*70)
