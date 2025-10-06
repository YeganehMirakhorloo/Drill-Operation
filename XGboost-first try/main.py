"""
================================================================================
DRILLING OPTIMIZATION SYSTEM - XGBoost Version
================================================================================
This system trains machine learning models to predict drilling parameters
and optimizes drilling operations using XGBoost and Differential Evolution.

Key Features:
- Separate training and testing datasets
- Baseline models (Ridge, Random Forest)
- XGBoost models with hyperparameter optimization
- Drilling parameter optimization (WOB, RPM)
- Comprehensive reporting and visualization

Author: Claude Code
Date: 2025/10/05 (1404/07/13)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split

# Import custom modules
from baseline_models import train_baseline_models
from drilling_xgboost import DrillingXGBoostPredictor  # ⭐ FIXED: Changed from xgboost_predictor
from xgboost_hyperparameter_optimizer import optimize_xgboost_hyperparameters
from de_optimizer import DifferentialEvolution  # ⭐ FIXED: Changed from differential_evolution
# Import plotting utilities
from plot_utils import plot_both_models_comparison, plot_prediction_comparison

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def clean_numeric_column(series):
    """
    Clean numeric column by converting strings to floats
    
    Args:
        series: Pandas Series to clean
        
    Returns:
        Cleaned Series with numeric values
    """
    def convert_to_float(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove whitespace and common non-numeric characters
            cleaned = value.strip().replace(',', '').replace(' ', '')
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        return np.nan
    
    return series.apply(convert_to_float)


def load_drilling_data(file_path, columns_to_drop=None, verbose=True):
    """
    Load and preprocess drilling data from Excel file with automatic header detection
    
    Args:
        file_path: Path to Excel file
        columns_to_drop: List of column names to drop (optional)
        verbose: Print detailed information about data loading
        
    Returns:
        Cleaned DataFrame with standardized column names
    """
    try:
        if verbose:
            print(f"\n📂 Loading data from: {file_path}")
            print(f"{'='*80}")
        
        # Step 1: Auto-detect header row
        temp_df = pd.read_excel(file_path, header=None, nrows=10)
        header_row = 0
        
        for idx, row in temp_df.iterrows():
            row_str = ' '.join(row.astype(str).values).lower()
            if any(keyword in row_str for keyword in ['wob', 'rpm', 'spp', 'rop', 'depth']):
                header_row = idx
                break
        
        if verbose:
            print(f"✓ Header detected at row: {header_row}")
        
        # Step 2: Load data with detected header
        data = pd.read_excel(file_path, header=header_row)
        
        if verbose:
            print(f"✓ Initial shape: {data.shape}")
            print(f"  Columns found: {list(data.columns)}")
        
        # Step 3: Drop specified columns if provided
        if columns_to_drop:
            existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
            if existing_cols_to_drop:
                data = data.drop(columns=existing_cols_to_drop)
                if verbose:
                    print(f"\n🗑️  Dropped columns: {existing_cols_to_drop}")
        
        # Step 4: Standardize column names
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower().strip()
            
            # Map common variations to standard names
            if 'wob' in col_lower or 'weight on bit' in col_lower:
                column_mapping[col] = 'WOB'
            elif col_lower in ['rpm', 'rotary speed', 'rotation']:
                column_mapping[col] = 'RPM'
            elif 'spp' in col_lower or 'standpipe pressure' in col_lower or 'pump pressure' in col_lower:
                column_mapping[col] = 'SPP'
            elif col_lower in ['q', 'flow rate', 'pump rate', 'flow']:
                column_mapping[col] = 'Q'
            elif 'depth' in col_lower or 'md' in col_lower:
                column_mapping[col] = 'Depth'
            elif 'rop' in col_lower or 'rate of penetration' in col_lower:
                column_mapping[col] = 'ROP'
            elif 'torque' in col_lower and 'surface' in col_lower:
                column_mapping[col] = 'Surface_Torque'
            elif 'hook' in col_lower and 'load' in col_lower:
                column_mapping[col] = 'Hook_Load'
            elif 'viscosity' in col_lower or 'vis' in col_lower:
                column_mapping[col] = 'Viscosity'
            elif col_lower in ['mw', 'mud weight', 'density']:
                column_mapping[col] = 'Mw'
        
        data = data.rename(columns=column_mapping)
        
        if verbose:
            print(f"\n📝 Standardized column names:")
            for old, new in column_mapping.items():
                print(f"  '{old}' → '{new}'")
        
        # Step 5: Check for required columns
        required_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'ROP', 'Surface_Torque']
        optional_columns = ['Hook_Load', 'Viscosity', 'Mw']
        
        available_required = [col for col in required_columns if col in data.columns]
        missing_required = [col for col in required_columns if col not in data.columns]
        available_optional = [col for col in optional_columns if col in data.columns]
        
        if verbose:
            print(f"\n✅ Column Status:")
            print(f"  Required columns available: {available_required}")
            if missing_required:
                print(f"  ⚠️  Missing required columns: {missing_required}")
            if available_optional:
                print(f"  Optional columns available: {available_optional}")
        
        # Step 6: Handle Hook_Load specially
        if 'Hook_Load' not in data.columns:
            if 'Viscosity' in data.columns:
                data['Hook_Load'] = data['Viscosity']
                if verbose:
                    print(f"\n🔄 Using 'Viscosity' as substitute for 'Hook_Load'")
            elif 'Mw' in data.columns:
                data['Hook_Load'] = data['Mw']
                if verbose:
                    print(f"\n🔄 Using 'Mw' as substitute for 'Hook_Load'")
            else:
                raise ValueError("Missing 'Hook_Load' and no suitable substitute (Viscosity or Mw) found!")
        
        # Step 7: Clean all available numeric columns
        numeric_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'ROP', 'Surface_Torque', 'Hook_Load']
        available_numeric = [col for col in numeric_columns if col in data.columns]
        
        if verbose:
            print(f"\n🧹 Cleaning numeric columns...")
        
        cleaning_stats = {}
        for col in available_numeric:
            original_values = data[col].copy()
            data[col] = clean_numeric_column(data[col])
            
            # Calculate cleaning statistics
            original_valid = original_values.notna().sum()
            cleaned_valid = data[col].notna().sum()
            cleaning_stats[col] = {
                'before': original_valid,
                'after': cleaned_valid,
                'removed': original_valid - cleaned_valid
            }
        
        if verbose:
            print(f"\n📊 Cleaning Statistics:")
            for col, stats in cleaning_stats.items():
                if stats['removed'] > 0:
                    print(f"  {col}: {stats['before']} → {stats['after']} "
                          f"(removed {stats['removed']} invalid values)")
        
        # Step 8: Select final columns
        final_columns = [col for col in required_columns if col in data.columns]
        
        # Step 9: Verify minimum required columns
        if len(final_columns) < 5:
            print(f"\n⚠️  WARNING: Only {len(final_columns)} columns available. Minimum 5 required!")
            print(f"  Available: {final_columns}")
            print(f"  Missing: {[c for c in required_columns if c not in final_columns]}")
        
        # Step 10: Remove rows with missing values
        data_clean = data[final_columns].copy()
        initial_rows = len(data_clean)
        data_clean = data_clean.dropna()
        final_rows = len(data_clean)
        
        if verbose:
            dropped_rows = initial_rows - final_rows
            print(f"\n🧹 Removed {dropped_rows} rows with missing values")
            print(f"  Final dataset: {final_rows} rows × {len(final_columns)} columns")
        
        # Step 11: Final verification
        if verbose:
            print(f"\n✅ Data loading completed successfully!")
            print(f"  Shape: {data_clean.shape}")
            print(f"  Columns: {list(data_clean.columns)}")
            print(f"  Memory usage: {data_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return data_clean
    
    except Exception as e:
        print(f"\n❌ Error loading data: {str(e)}")
        raise


def train_models_with_separate_data(train_data, test_data, optimize_hyperparams=False):
    """
    Train models using separate training and testing datasets
    Works with available columns only
    
    Args:
        train_data: DataFrame with training data
        test_data: DataFrame with testing data
        optimize_hyperparams: Whether to optimize XGBoost hyperparameters
        
    Returns:
        Dictionary containing trained models and results
    """
    print(f"\n{'='*80}")
    print("📊 Analyzing Available Columns")
    print(f"{'='*80}")
    
    # Define all possible feature columns
    all_possible_features = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    all_possible_targets = ['ROP', 'Surface_Torque']
    
    # Find available features in training data
    available_features = [col for col in all_possible_features if col in train_data.columns]
    available_targets = [col for col in all_possible_targets if col in train_data.columns]
    
    print(f"\n✓ Available features ({len(available_features)}):")
    for feat in available_features:
        print(f"  • {feat}")
    
    print(f"\n✓ Available targets ({len(available_targets)}):")
    for tgt in available_targets:
        print(f"  • {tgt}")
    
    # Check minimum requirements
    if len(available_features) < 3:
        raise ValueError(f"❌ Need at least 3 features, found only {len(available_features)}: {available_features}")
    
    if len(available_targets) == 0:
        raise ValueError(f"❌ Need at least 1 target (ROP or Surface_Torque), found none")
    
    # Verify test data has same columns
    test_features = [col for col in available_features if col in test_data.columns]
    test_targets = [col for col in available_targets if col in test_data.columns]
    
    if test_features != available_features:
        raise ValueError(f"❌ Test data missing features: {set(available_features) - set(test_features)}")
    
    if test_targets != available_targets:
        raise ValueError(f"❌ Test data missing targets: {set(available_targets) - set(test_targets)}")
    
    # Prepare training data
    X_train = train_data[available_features].values
    X_test = test_data[available_features].values
    
    print(f"\n{'='*80}")
    print(f"📊 Dataset Information:")
    print(f"  Training set: {len(X_train)} samples × {len(available_features)} features")
    print(f"  Testing set:  {len(X_test)} samples × {len(available_features)} features")
    print(f"  Targets:      {', '.join(available_targets)}")
    print(f"{'='*80}")
    
    results = {
        'features': available_features,
        'targets': available_targets,
        'models': {}
    }
    
    # ====== STEP 1: Train Baseline Models ======
    print(f"\n{'='*80}")
    print("STEP 1: Training Baseline Models (Ridge & Random Forest)")
    print(f"{'='*80}")
    
    try:
        baseline_results = train_baseline_models(train_data)
        results['baseline'] = baseline_results
    except Exception as e:
        print(f"⚠️  Warning: Baseline models failed: {e}")
        results['baseline'] = None
    
    # ====== STEP 2: Train XGBoost Models for Each Available Target ======
    print(f"\n{'='*80}")
    print("STEP 2: Training XGBoost Models")
    print(f"{'='*80}")
    
    for target_name in available_targets:
        print(f"\n🎯 Training XGBoost Model for: {target_name}")
        print(f"{'='*80}")
        
        # Get target data
        y_train = train_data[target_name].values
        y_test = test_data[target_name].values
        
        # Optional: Hyperparameter optimization
        best_params = None
        if optimize_hyperparams:
            print(f"\n⚙️  Optimizing hyperparameters for {target_name}...")
            
            # Split training data for validation
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            try:
                best_params, opt_history = optimize_xgboost_hyperparameters(
                    X_train_opt, y_train_opt,
                    X_val_opt, y_val_opt,
                    pop_size=20,
                    max_iter=30
                )
                
                results[f'{target_name}_optimization'] = {
                    'params': best_params,
                    'history': opt_history
                }
                
                print(f"✓ Optimization complete!")
                print(f"  Best params: {best_params}")
                
            except Exception as e:
                print(f"⚠️  Optimization failed: {e}")
                print(f"  Using default parameters")
        
        # Initialize model with optimized or default parameters
        model = DrillingXGBoostPredictor(
            n_estimators=int(best_params['n_estimators']) if best_params else 100,
            max_depth=int(best_params['max_depth']) if best_params else 6,
            learning_rate=best_params['learning_rate'] if best_params else 0.1,
            subsample=best_params['subsample'] if best_params else 0.8,
            colsample_bytree=best_params['colsample_bytree'] if best_params else 0.8,
            gamma=best_params['gamma'] if best_params else 0,
            min_child_weight=best_params['min_child_weight'] if best_params else 1,
            reg_alpha=best_params['reg_alpha'] if best_params else 0,
            reg_lambda=best_params['reg_lambda'] if best_params else 1
        )
        
        # Split training data for validation during training
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model (removed early_stopping_rounds and verbose parameters)
        print(f"\n🏋️  Training {target_name} model...")
        model.train(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # Evaluate on test set
        print(f"\n📊 Evaluating {target_name} model on test set...")
        test_results = model.evaluate(X_test, y_test)
        
        print(f"\n✅ {target_name} Model Results:")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  R²:   {test_results['r2']:.4f}")
        print(f"  AARE: {test_results['aare']:.2f}%")
        
        # Store model and results
        results['models'][target_name] = {
            'model': model,
            'test_results': test_results,
            'best_params': best_params
        }
    
    print(f"\n{'='*80}")
    print("✅ All models trained successfully!")
    print(f"{'='*80}")
    # ====== STEP 3: Generate Prediction Comparison Plots ======
    print(f"\n{'='*80}")
    print("STEP 3: Generating Prediction Comparison Plots")
    print(f"{'='*80}")
    
    
    # Prepare data for plotting
    plot_data = {}
    
    for target_name in available_targets:
        if target_name in results['models']:
            model = results['models'][target_name]['model']
            test_results = results['models'][target_name]['test_results']
            
            # Get training predictions
            y_train_actual = train_data[target_name].values
            y_train_pred = model.predict(X_train)
            
            # Get test predictions (already available)
            y_test_actual = test_results['y_true']
            y_test_pred = test_results['predictions']
            
            # Store for plotting
            plot_data[target_name] = {
                'train_actual': y_train_actual,
                'train_pred': y_train_pred,
                'test_actual': y_test_actual,
                'test_pred': y_test_pred
            }
    
    # Generate plots based on available targets
    if 'ROP' in plot_data and 'Surface_Torque' in plot_data:
        print("\n📊 Generating combined plot for ROP and Torque...")
        plot_both_models_comparison(
            plot_data['ROP']['train_actual'], plot_data['ROP']['train_pred'],
            plot_data['ROP']['test_actual'], plot_data['ROP']['test_pred'],
            plot_data['Surface_Torque']['train_actual'], plot_data['Surface_Torque']['train_pred'],
            plot_data['Surface_Torque']['test_actual'], plot_data['Surface_Torque']['test_pred'],
            save_path='results_comparison_xgboost.png'
        )
    else:
        # Plot individual models
        if 'ROP' in plot_data:
            print("\n📊 Generating ROP prediction plot...")
            plot_prediction_comparison(
                plot_data['ROP']['train_actual'], plot_data['ROP']['train_pred'],
                plot_data['ROP']['test_actual'], plot_data['ROP']['test_pred'],
                target_name='ROP',
                save_path='results_comparison_rop_xgboost.png'
            )
        
        if 'Surface_Torque' in plot_data:
            print("\n📊 Generating Torque prediction plot...")
            plot_prediction_comparison(
                plot_data['Surface_Torque']['train_actual'], plot_data['Surface_Torque']['train_pred'],
                plot_data['Surface_Torque']['test_actual'], plot_data['Surface_Torque']['test_pred'],
                target_name='Torque',
                save_path='results_comparison_torque_xgboost.png'
            )
    
    # Store plot data in results
    results['plot_data'] = plot_data
    
    print(f"\n{'='*80}")
    print("✅ All models trained successfully!")
    print(f"{'='*80}")

    return results



def generate_report(results, output_dir='results'):
    """
    Generate comprehensive report with visualizations

    Args:
        results: Dictionary containing all training results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("📊 Generating Report")
    print(f"{'='*80}")

    # Get available targets
    available_targets = results.get('targets', [])

    if not available_targets:
        print("⚠️  No targets available for reporting")
        return

    # ====== Plot 1: Model Comparison ======
    if results.get('baseline') and results.get('models'):
        fig, axes = plt.subplots(1, len(available_targets), figsize=(7*len(available_targets), 6))

        if len(available_targets) == 1:
            axes = [axes]

        for idx, target in enumerate(available_targets):
            ax = axes[idx]

            models = []
            r2_scores = []

            # Determine target key for baseline results
            if target == 'ROP':
                target_key = 'rop'
            elif target == 'Surface_Torque':
                target_key = 'torque'
            else:
                target_key = target.lower()

            # Get baseline results if available
            if results['baseline'] and target_key in results['baseline']:
                baseline = results['baseline'][target_key]
                models.extend(['Ridge', 'RandomForest'])
                r2_scores.extend([
                    baseline['ridge_r2'],  # ✅ FIXED: Correct key structure
                    baseline['rf_r2']       # ✅ FIXED: Correct key structure
                ])

            # Get XGBoost results
            if target in results['models']:
                models.append('XGBoost')
                r2_scores.append(results['models'][target]['test_results']['r2'])

            # Plot
            colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(models)]
            bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{target} Prediction - Model Comparison',
                        fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: model_comparison.png")
        plt.close()

    # ====== Plot 2: Optimization History (if available) ======
    has_optimization = any(f'{target}_optimization' in results for target in available_targets)

    if has_optimization:
        fig, axes = plt.subplots(1, len(available_targets), figsize=(7*len(available_targets), 6))

        if len(available_targets) == 1:
            axes = [axes]

        for idx, target in enumerate(available_targets):
            ax = axes[idx]
            opt_key = f'{target}_optimization'

            if opt_key in results:
                history = results[opt_key]['history']
                
                # Plot best fitness
                if 'best_fitness' in history:
                    generations = range(1, len(history['best_fitness']) + 1)
                    ax.plot(generations, history['best_fitness'], 
                           'b-', linewidth=2, label='Best R²')
                
                # Plot mean fitness if available
                if 'mean_fitness' in history:
                    ax.plot(generations, history['mean_fitness'], 
                           'r--', linewidth=1.5, alpha=0.6, label='Mean R²')
                
                ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
                ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
                ax.set_title(f'{target} - Hyperparameter Optimization',
                            fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: optimization_history.png")
        plt.close()
    
    # ====== Plot 3: Predictions vs Actual ======
    # Check if plots were already generated during training
    if 'plot_data' in results:
        print(f"✓ Prediction comparison plots already generated during training")
        print(f"  - Skipping duplicate plot generation")
    elif results.get('models'):
        # Generate plots only if not already created
        print(f"\n📊 Generating prediction vs actual plots...")
        
        fig, axes = plt.subplots(1, len(available_targets), figsize=(7*len(available_targets), 6))

        if len(available_targets) == 1:
            axes = [axes]

        for idx, target in enumerate(available_targets):
            if target in results['models']:
                ax = axes[idx]
                test_results = results['models'][target]['test_results']

                y_true = test_results.get('y_true', [])
                predictions = test_results.get('predictions', [])

                # Scatter plot
                ax.scatter(y_true, predictions, alpha=0.5, s=20)

                # Perfect prediction line
                min_val = min(min(y_true) if len(y_true) > 0 else 0,
                            min(predictions) if len(predictions) > 0 else 0)
                max_val = max(max(y_true) if len(y_true) > 0 else 1,
                            max(predictions) if len(predictions) > 0 else 1)
                ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Perfect Prediction')

                ax.set_xlabel(f'Actual {target}', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Predicted {target}', fontsize=12, fontweight='bold')
                ax.set_title(f'{target} - Predictions vs Actual\nR² = {test_results["r2"]:.4f}',
                            fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: predictions_vs_actual.png")
        plt.close()


    # ====== Print Summary Report ======
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    for target in available_targets:
        print(f"\n🎯 {target} Prediction:")
        print(f"{'─'*80}")

        # Determine target key for baseline
        if target == 'ROP':
            target_key = 'rop'
        elif target == 'Surface_Torque':
            target_key = 'torque'
        else:
            target_key = target.lower()

        # Baseline results
        if results.get('baseline') and target_key in results['baseline']:
            baseline = results['baseline'][target_key]
            print(f"\n  Baseline Models:")
            print(f"    Ridge Regression:")
            print(f"      R² = {baseline['ridge_r2']:.4f}, RMSE = {baseline['ridge_rmse']:.4f}")
            print(f"    Random Forest:")
            print(f"      R² = {baseline['rf_r2']:.4f}, RMSE = {baseline['rf_rmse']:.4f}")

        # XGBoost results
        if target in results.get('models', {}):
            xgb_results = results['models'][target]['test_results']
            print(f"\n  XGBoost Model:")
            print(f"    R²   = {xgb_results['r2']:.4f}")
            print(f"    RMSE = {xgb_results['rmse']:.4f}")
            print(f"    AARE = {xgb_results['aare']:.2f}%")

            # Best parameters if available
            if results['models'][target].get('best_params'):
                print(f"\n  Optimized Hyperparameters:")
                for param, value in results['models'][target]['best_params'].items():
                    print(f"    {param}: {value}")

    print(f"\n{'='*80}")
    print("✅ Report generation completed!")
    print(f"{'='*80}\n")



def main():
    """
    Main execution function
    """
    print(f"\n{'='*80}")
    print("🚀 DRILLING OPTIMIZATION SYSTEM - XGBoost Version")
    print(f"{'='*80}")
    
    # Configuration
    TRAIN_FILE = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#RR#34.xlsx"
    TEST_FILE = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#MI#131.xlsx"
    OPTIMIZE_HYPERPARAMS = False  # Set to True to enable optimization
    
    # Load data
    print("\n📂 Loading datasets...")
    train_data = load_drilling_data(TRAIN_FILE, verbose=True)
    test_data = load_drilling_data(TEST_FILE, verbose=True)
    
    # Train models
    results = train_models_with_separate_data(
        train_data,
        test_data,
        optimize_hyperparams=OPTIMIZE_HYPERPARAMS
    )
    
    # Generate report
    generate_report(results)
    
    print(f"\n{'='*80}")
    print("✅ SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
