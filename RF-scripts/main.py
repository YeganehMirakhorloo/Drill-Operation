"""
================================================================================
DRILLING OPTIMIZATION SYSTEM - XGBoost & Random Forest Version
================================================================================
This system trains machine learning models to predict drilling parameters
and optimizes drilling operations using XGBoost/Random Forest and Differential Evolution.

Key Features:
- Separate training and testing datasets with manual header control
- Baseline models (Ridge, Random Forest)
- XGBoost/Random Forest models with hyperparameter optimization
- Drilling parameter optimization (WOB, RPM)
- Comprehensive reporting and visualization

Author: Claude Code
Date: 2025/10/06 (1404/07/14)
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
# from drilling_xgboost import DrillingXGBoostPredictor
from drilling_random_forest import DrillingRandomForestPredictor
# from xgboost_hyperparameter_optimizer import optimize_xgboost_hyperparameters
from rf_hyperparameter_optimizer import optimize_rf_hyperparameters
from de_optimizer import DifferentialEvolution

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

def load_drilling_data(file_path, header_row=0, columns_to_drop=None, verbose=True):
    """
    Load and preprocess drilling data from Excel file with MANUAL header specification

    Args:
        file_path: Path to Excel file
        header_row: Integer specifying which row contains the headers (0-based index)
                   - If your headers are in row 1 (Excel), use header_row=0
                   - If your headers are in row 3 (Excel), use header_row=2
                   - Set to None if no headers exist
        columns_to_drop: List of column names to drop (optional)
        verbose: Print detailed information about data loading

    Returns:
        Cleaned DataFrame with standardized column names
    
    Examples:
        # Headers in first row
        data = load_drilling_data('file.xlsx', header_row=0)
        
        # Headers in third row (skip first 2 rows)
        data = load_drilling_data('file.xlsx', header_row=2)
        
        # No headers, will use column indices
        data = load_drilling_data('file.xlsx', header_row=None)
    """
    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"📂 LOADING DATA")
            print(f"{'='*80}")
            print(f"  File: {file_path}")
            print(f"  Header row: {header_row if header_row is not None else 'None (no headers)'}")

        # Step 1: Load data with specified header row
        if header_row is not None:
            data = pd.read_excel(file_path, header=header_row)
            if verbose:
                print(f"  ✓ Loaded with header at row {header_row}")
        else:
            data = pd.read_excel(file_path, header=None)
            # Generate column names if no header
            data.columns = [f'Column_{i}' for i in range(len(data.columns))]
            if verbose:
                print(f"  ✓ Loaded without headers, generated column names")

        if verbose:
            print(f"\n📊 Initial Data Info:")
            print(f"  Shape: {data.shape}")
            print(f"  Columns ({len(data.columns)}): {list(data.columns)}")
            
            # Show first few rows to help user verify
            print(f"\n  First 3 rows preview:")
            print(data.head(3).to_string(index=True))

        # Step 2: Drop specified columns if provided
        if columns_to_drop:
            existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
            if existing_cols_to_drop:
                data = data.drop(columns=existing_cols_to_drop)
                if verbose:
                    print(f"\n🗑️  Dropped columns: {existing_cols_to_drop}")
                    print(f"  Remaining columns: {list(data.columns)}")

        # Step 3: Standardize column names
        if verbose:
            print(f"\n📝 COLUMN NAME STANDARDIZATION:")
            print(f"{'='*80}")
        
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower().strip()
            original_col = col

            # Map common variations to standard names
            if 'wob' in col_lower or 'weight on bit' in col_lower or 'weight_on_bit' in col_lower:
                column_mapping[col] = 'WOB'
            elif col_lower in ['rpm', 'rotary speed', 'rotation', 'rotary_speed']:
                column_mapping[col] = 'RPM'
            elif 'spp' in col_lower or 'standpipe' in col_lower or 'pump pressure' in col_lower or 'pump_pressure' in col_lower:
                column_mapping[col] = 'SPP'
            elif col_lower in ['q', 'gpm', 'flow rate', 'pump rate', 'flow', 'flow_rate', 'pump_rate']:
                column_mapping[col] = 'Q'
            elif 'depth' in col_lower or 'md' in col_lower or 'measured_depth' in col_lower:
                column_mapping[col] = 'Depth'
            elif 'rop' in col_lower or 'rate of penetration' in col_lower or 'rate_of_penetration' in col_lower:
                column_mapping[col] = 'ROP'
            elif ('torque' in col_lower and 'surface' in col_lower) or 'surface_torque' in col_lower:
                column_mapping[col] = 'Surface_Torque'
            elif ('hook' in col_lower and 'load' in col_lower) or 'hook_load' in col_lower:
                column_mapping[col] = 'Hook_Load'
            elif 'viscosity' in col_lower or col_lower == 'vis':
                column_mapping[col] = 'Viscosity'
            elif col_lower in ['mw', 'mud weight', 'density', 'mud_weight']:
                column_mapping[col] = 'Mw'

        data = data.rename(columns=column_mapping)

        if verbose:
            if column_mapping:
                print(f"  Standardized {len(column_mapping)} column(s):")
                for old, new in column_mapping.items():
                    print(f"    '{old}' → '{new}'")
            else:
                print(f"  ⚠️  No columns matched standard patterns")
                print(f"  Current columns: {list(data.columns)}")

        # Step 4: Check for required columns
        required_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'ROP', 'Surface_Torque']
        optional_columns = ['Hook_Load', 'Viscosity', 'Mw']

        available_required = [col for col in required_columns if col in data.columns]
        missing_required = [col for col in required_columns if col not in data.columns]
        available_optional = [col for col in optional_columns if col in data.columns]

        if verbose:
            print(f"\n✅ COLUMN STATUS:")
            print(f"{'='*80}")
            print(f"  Required columns ({len(available_required)}/{len(required_columns)}):")
            for col in available_required:
                print(f"    ✓ {col}")
            
            if missing_required:
                print(f"\n  ⚠️  Missing required columns ({len(missing_required)}):")
                for col in missing_required:
                    print(f"    ✗ {col}")
            
            if available_optional:
                print(f"\n  Optional columns ({len(available_optional)}):")
                for col in available_optional:
                    print(f"    ✓ {col}")

        # Step 5: Handle Hook_Load specially
        if 'Hook_Load' not in data.columns:
            if 'Viscosity' in data.columns:
                data['Hook_Load'] = data['Viscosity']
                if verbose:
                    print(f"\n🔄 SUBSTITUTION: Using 'Viscosity' as 'Hook_Load'")
            elif 'Mw' in data.columns:
                data['Hook_Load'] = data['Mw']
                if verbose:
                    print(f"\n🔄 SUBSTITUTION: Using 'Mw' as 'Hook_Load'")
            else:
                raise ValueError(
                    "\n❌ ERROR: Missing 'Hook_Load' and no suitable substitute!\n"
                    f"   Available columns: {list(data.columns)}\n"
                    "   Need either: Hook_Load, Viscosity, or Mw"
                )

        # Step 6: Clean all available numeric columns
        numeric_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'ROP', 'Surface_Torque', 'Hook_Load']
        available_numeric = [col for col in numeric_columns if col in data.columns]

        if verbose:
            print(f"\n🧹 CLEANING NUMERIC DATA:")
            print(f"{'='*80}")

        cleaning_stats = {}
        for col in available_numeric:
            original_values = data[col].copy()
            data[col] = clean_numeric_column(data[col])

            # Calculate cleaning statistics
            original_valid = original_values.notna().sum()
            cleaned_valid = data[col].notna().sum()
            removed = original_valid - cleaned_valid
            
            cleaning_stats[col] = {
                'before': original_valid,
                'after': cleaned_valid,
                'removed': removed
            }

        if verbose:
            for col, stats in cleaning_stats.items():
                status = "✓" if stats['removed'] == 0 else "⚠️"
                print(f"  {status} {col}: {stats['before']} → {stats['after']} ", end='')
                if stats['removed'] > 0:
                    print(f"(removed {stats['removed']})")
                else:
                    print(f"(clean)")

        # Step 7: Select final columns
        final_columns = [col for col in required_columns if col in data.columns]

        # Step 8: Verify minimum required columns
        if len(final_columns) < 5:
            raise ValueError(
                f"\n❌ ERROR: Insufficient columns!\n"
                f"   Found {len(final_columns)}: {final_columns}\n"
                f"   Need at least 5 from: {required_columns}\n"
                f"   Missing: {[c for c in required_columns if c not in final_columns]}"
            )

        # Step 9: Remove rows with missing values
        data_clean = data[final_columns].copy()
        initial_rows = len(data_clean)
        data_clean = data_clean.dropna()
        final_rows = len(data_clean)

        if verbose:
            dropped_rows = initial_rows - final_rows
            print(f"\n🧹 DATA CLEANING SUMMARY:")
            print(f"{'='*80}")
            print(f"  Initial rows:    {initial_rows}")
            print(f"  Dropped rows:    {dropped_rows} ({dropped_rows/initial_rows*100:.1f}%)")
            print(f"  Final rows:      {final_rows}")
            print(f"  Final columns:   {len(final_columns)}")

        # Step 10: Final verification and statistics
        if verbose:
            print(f"\n📊 FINAL DATASET SUMMARY:")
            print(f"{'='*80}")
            print(f"  Shape: {data_clean.shape[0]} rows × {data_clean.shape[1]} columns")
            print(f"  Columns: {list(data_clean.columns)}")
            print(f"  Memory: {data_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            print(f"\n  Data Statistics:")
            print(data_clean.describe().to_string())
            
            print(f"\n{'='*80}")
            print(f"✅ DATA LOADING COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")

        return data_clean

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR LOADING DATA")
        print(f"{'='*80}")
        print(f"  File: {file_path}")
        print(f"  Error: {str(e)}")
        print(f"{'='*80}\n")
        raise

def train_models_with_separate_data(train_data, test_data, model_type='xgboost', optimize_hyperparams=False):
    """
    Train models using separate training and testing datasets
    Works with available columns only

    Args:
        train_data: DataFrame with training data
        test_data: DataFrame with testing data
        model_type: 'xgboost' or 'rf' (random forest)
        optimize_hyperparams: Whether to optimize hyperparameters

    Returns:
        Dictionary containing trained models and results
    """
    print(f"\n{'='*80}")
    print(f"🤖 MODEL TRAINING - {model_type.upper()}")
    print(f"{'='*80}")

    # Define all possible feature columns
    all_possible_features = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    all_possible_targets = ['ROP', 'Surface_Torque']

    # Find available features in training data
    available_features = [col for col in all_possible_features if col in train_data.columns]
    available_targets = [col for col in all_possible_targets if col in train_data.columns]

    print(f"\n📋 Available Features ({len(available_features)}):")
    for feat in available_features:
        print(f"  • {feat}")

    print(f"\n🎯 Available Targets ({len(available_targets)}):")
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
    print(f"📊 DATASET INFORMATION:")
    print(f"{'='*80}")
    print(f"  Training set:  {len(X_train):>6} samples × {len(available_features)} features")
    print(f"  Testing set:   {len(X_test):>6} samples × {len(available_features)} features")
    print(f"  Model type:    {model_type.upper()}")
    print(f"  Optimization:  {'ON' if optimize_hyperparams else 'OFF'}")
    print(f"{'='*80}")

    results = {
        'features': available_features,
        'targets': available_targets,
        'model_type': model_type,
        'models': {}
    }

    # ====== STEP 1: Train Baseline Models ======
    print(f"\n{'='*80}")
    print("STEP 1: BASELINE MODELS (Ridge & Random Forest)")
    print(f"{'='*80}")

    try:
        baseline_results = train_baseline_models(train_data)
        results['baseline'] = baseline_results
    except Exception as e:
        print(f"⚠️  Warning: Baseline models failed: {e}")
        results['baseline'] = None

    # ====== STEP 2: Train Models for Each Available Target ======
    print(f"\n{'='*80}")
    print(f"STEP 2: {model_type.upper()} MODELS")
    print(f"{'='*80}")

    for target_name in available_targets:
        print(f"\n🎯 Training {model_type.upper()} Model for: {target_name}")
        print(f"{'-'*80}")

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

            # if model_type == 'xgboost':
            #     best_params, opt_history = optimize_xgboost_hyperparameters(
            #         X_train_opt, y_train_opt, X_val_opt, y_val_opt,
            #         pop_size=15, max_iter=20
            #     )
            #     results[f'{target_name}_optimization'] = {
            #         'params': best_params,
            #         'history': opt_history
            #     }
            # elif 
            if model_type == 'rf':
                best_params, opt_history = optimize_rf_hyperparameters(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                    pop_size=15, max_iter=20
                )
                results[f'{target_name}_optimization'] = {
                    'params': best_params,
                    'history': opt_history
                }

        # Create model with best params or defaults
        # if model_type == 'xgboost':
        #     if best_params:
        #         model = DrillingXGBoostPredictor(**best_params)
        #     else:
        #         model = DrillingXGBoostPredictor(
        #             n_estimators=100,
        #             max_depth=6,
        #             learning_rate=0.1,
        #             subsample=0.8,
        #             colsample_bytree=0.8
        #         )
        # elif 
        if model_type == 'rf':
            if best_params:
                model = DrillingRandomForestPredictor(**best_params)
            else:
                model = DrillingRandomForestPredictor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt'
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'xgboost' or 'rf'")

        # Train on full dataset
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        model.train(X_train_split, y_train_split, X_val_split, y_val_split)

        # Evaluate
        train_results = model.evaluate(X_train, y_train)
        test_results = model.evaluate(X_test, y_test)

        print(f"\n📊 {target_name} Model Performance:")
        print(f"  {'Dataset':<10} {'R²':>8} {'RMSE':>10} {'AARE':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Train':<10} {train_results['r2']:>8.4f} {train_results['rmse']:>10.4f} {train_results['aare']:>9.2f}%")
        print(f"  {'Test':<10} {test_results['r2']:>8.4f} {test_results['rmse']:>10.4f} {test_results['aare']:>9.2f}%")

        # Store results
        results['models'][target_name] = {
            'model': model,
            'train': train_results,
            'test': test_results,
            'feature_importance': model.get_feature_importance()
        }

    return results

def generate_report(results, output_dir='results'):
    """
    Generate comprehensive report with plots and tables

    Args:
        results: Dictionary containing all model results
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("📊 GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")

    model_type = results.get('model_type', 'unknown')
    
    # ====== 1. Performance Summary Table ======
    print(f"\n1️⃣  Performance Summary ({model_type.upper()}):")
    print(f"{'='*80}")

    summary_data = []
    for target in results['targets']:
        if target in results['models']:
            model_results = results['models'][target]
            summary_data.append({
                'Target': target,
                'Train R²': f"{model_results['train']['r2']:.4f}",
                'Test R²': f"{model_results['test']['r2']:.4f}",
                'Train RMSE': f"{model_results['train']['rmse']:.4f}",
                'Test RMSE': f"{model_results['test']['rmse']:.4f}",
                'Train AARE': f"{model_results['train']['aare']:.2f}%",
                'Test AARE': f"{model_results['test']['aare']:.2f}%"
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, f'performance_summary_{model_type}.csv'), index=False)
    print(f"\n✓ Saved: performance_summary_{model_type}.csv")

    # ====== 2. Baseline Comparison (if available) ======
    if results.get('baseline'):
        print(f"\n2️⃣  Baseline Comparison:")
        print(f"{'='*80}")

        baseline = results['baseline']
        comparison_data = []

        for target_key in ['rop', 'torque']:
            if target_key in baseline:
                target_name = 'ROP' if target_key == 'rop' else 'Surface_Torque'
                if target_name in results['models']:
                    comparison_data.append({
                        'Target': target_name,
                        'Ridge R²': f"{baseline[target_key]['ridge_r2']:.4f}",
                        'Baseline RF R²': f"{baseline[target_key]['rf_r2']:.4f}",
                        f'{model_type.upper()} R²': f"{results['models'][target_name]['test']['r2']:.4f}"
                    })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            comparison_df.to_csv(os.path.join(output_dir, f'baseline_comparison_{model_type}.csv'), index=False)
            print(f"\n✓ Saved: baseline_comparison_{model_type}.csv")

    # ====== 3. Feature Importance Plot ======
    print(f"\n3️⃣  Generating Feature Importance Plot...")

    n_targets = len(results['targets'])
    fig, axes = plt.subplots(1, n_targets, figsize=(8 * n_targets, 6))
    if n_targets == 1:
        axes = [axes]

    for idx, target in enumerate(results['targets']):
        if target in results['models']:
            importance = results['models'][target]['feature_importance']
            features = results['features']

            # Sort by importance
            indices = np.argsort(importance)[::-1]

            axes[idx].barh(range(len(indices)), importance[indices], 
                          color='skyblue' if idx == 0 else 'lightcoral')
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([features[i] for i in indices])
            axes[idx].set_xlabel('Importance', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{target} - Feature Importance ({model_type.upper()})', 
                              fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{model_type}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: feature_importance_{model_type}.png")
    plt.close()

    print(f"\n✅ Report generation complete!")
    print(f"  Files saved in: {output_dir}/")

def main():
    """
    Main execution function
    """
    print(f"\n{'='*80}")
    print("🚀 DRILLING OPTIMIZATION SYSTEM")
    print(f"   XGBoost & Random Forest Version")
    print(f"{'='*80}")

    # ============================================================================
    # CONFIGURATION - ADJUST THESE SETTINGS
    # ============================================================================
    
    # File paths
    TRAIN_FILE = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#RR#34.xlsx"
    TEST_FILE = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#MI#131.xlsx"
    
    # ⭐ HEADER ROW SPECIFICATION (0-based index)
    # Examples:
    #   - If headers are in row 1 (Excel), use: 0
    #   - If headers are in row 3 (Excel), use: 2
    #   - If no headers exist, use: None
    TRAIN_HEADER_ROW = 0  # ⭐ CHANGE THIS for your training file
    TEST_HEADER_ROW = 2   # ⭐ CHANGE THIS for your testing file
    
    # Model configuration
    MODEL_TYPE = 'rf'  # ⭐ CHANGE THIS: 'xgboost' or 'rf'
    OPTIMIZE_HYPERPARAMS = False  # Set to True to enable optimization
    
    # Optional: Columns to drop (if any)
    COLUMNS_TO_DROP = None  # Example: ['Date', 'Time', 'Comments']
    
    # ============================================================================

    print(f"\n⚙️  CONFIGURATION:")
    print(f"{'='*80}")
    print(f"  Training file:        {os.path.basename(TRAIN_FILE)}")
    print(f"  Training header row:  {TRAIN_HEADER_ROW}")
    print(f"  Testing file:         {os.path.basename(TEST_FILE)}")
    print(f"  Testing header row:   {TEST_HEADER_ROW}")
    print(f"  Model type:           {MODEL_TYPE.upper()}")
    print(f"  Hyperparameter opt:   {'ON' if OPTIMIZE_HYPERPARAMS else 'OFF'}")
    print(f"{'='*80}")

    # Load data
    print("\n" + "="*80)
    print("📂 STEP 1: LOADING DATASETS")
    print("="*80)
    
    train_data = load_drilling_data(
        TRAIN_FILE, 
        header_row=TRAIN_HEADER_ROW,
        columns_to_drop=COLUMNS_TO_DROP,
        verbose=True
    )
    
    test_data = load_drilling_data(
        TEST_FILE, 
        header_row=TEST_HEADER_ROW,
        columns_to_drop=COLUMNS_TO_DROP,
        verbose=True
    )

    # Train models
    print("\n" + "="*80)
    print("📂 STEP 2: TRAINING MODELS")
    print("="*80)
    
    results = train_models_with_separate_data(
        train_data,
        test_data,
        model_type=MODEL_TYPE,
        optimize_hyperparams=OPTIMIZE_HYPERPARAMS
    )

    # Generate report
    print("\n" + "="*80)
    print("📂 STEP 3: GENERATING REPORTS")
    print("="*80)
    
    generate_report(results, output_dir=f'results_{MODEL_TYPE}')

    print(f"\n{'='*80}")
    print("✅ SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
