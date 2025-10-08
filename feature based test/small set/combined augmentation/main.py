import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re
from drilling_ann import DrillingPredictor
from data_augmentation import augment_training_data
from ann_hyperparameter_optimizer import optimize_ann_hyperparameters  # ⭐ IMPORT

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def convert_range_to_number(value):
    """
    Convert range strings like '15-25' to their AVERAGE (20.0)
    ⭐ ALWAYS returns float or np.nan (NEVER string) ⭐
    """
    # Handle DataFrame (happens with duplicate columns)
    if isinstance(value, pd.DataFrame):
        if len(value.columns) > 0:
            value = value.iloc[:, 0]
        else:
            return np.nan

    # Handle Series
    if isinstance(value, pd.Series):
        if len(value) > 0:
            value = value.iloc[0]
        else:
            return np.nan

    # Handle None, NaN, or NaT
    if value is None or pd.isna(value):
        return np.nan

    # If it's already numeric, return it
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    # Convert to string and clean
    value_str = str(value).strip().lower()

    # Empty or placeholder strings
    if not value_str or value_str in ['', '-', 'n/a', 'na', 'null', 'none', 'nan', 'an', 'mar']:
        return np.nan

    # Try direct float conversion first
    try:
        return float(value_str)
    except (ValueError, TypeError):
        pass

    # Pattern for ranges: "20-25", "20 - 25", "20to25", etc.
    range_patterns = [
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',      # "20-25" or "20 - 25"
        r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)',     # "20 to 25"
        r'(\d+(?:\.\d+)?)\s*~\s*(\d+(?:\.\d+)?)',      # "20 ~ 25"
        r'(\d+(?:\.\d+)?)\s*تا\s*(\d+(?:\.\d+)?)',     # "20 تا 25" (Persian)
    ]

    for pattern in range_patterns:
        match = re.search(pattern, value_str, re.IGNORECASE)
        if match:
            try:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                average = (min_val + max_val) / 2.0
                return average
            except (ValueError, TypeError):
                continue

    # Extract any number from the string (e.g., "25 kg" -> 25.0)
    number_match = re.search(r'(\d+(?:\.\d+)?)', value_str)
    if number_match:
        try:
            return float(number_match.group(1))
        except (ValueError, TypeError):
            pass

    # If all else fails, return NaN
    return np.nan


def clean_numeric_column(series):
    """
    Clean a pandas series containing mixed numeric and string data
    ⭐ GUARANTEED to return float64 dtype ⭐
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    # If it's already numeric, return as-is
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    # Apply conversion function to each element
    print(f"    Converting {series.name if hasattr(series, 'name') else 'column'}...", end='')

    # Apply our converter
    cleaned = series.apply(convert_range_to_number)

    # Double-check: convert to numeric
    cleaned_numeric = pd.to_numeric(cleaned, errors='coerce')

    # Verify dtype
    if not pd.api.types.is_numeric_dtype(cleaned_numeric):
        raise ValueError(f"❌ Failed to convert to numeric! Dtype: {cleaned_numeric.dtype}")

    # Count how many values were converted
    original_non_null = series.notna().sum()
    cleaned_non_null = cleaned_numeric.notna().sum()

    if original_non_null > 0:
        conversion_rate = 100 * cleaned_non_null / original_non_null
        print(f" OK ({cleaned_non_null}/{original_non_null} = {conversion_rate:.1f}%)")
    else:
        print(f" OK (all NaN)")

    return cleaned_numeric


def detect_header_row(file_path, max_rows_to_check=20):
    """
    🔍 Automatically detect which row contains the header
    """
    print(f"\n🔍 Detecting header row...")

    drilling_keywords = [
        'depth', 'wob', 'mob', 'rop', 'rpm', 'torque', 'gpm', 'spp',
        'mw', 'vis', 'flow', 'pressure', 'rate', 'bit', 'hole',
        'md', 'tvd', 'inc', 'azi', 'dls'
    ]

    best_header_row = None
    max_score = 0

    for row_idx in range(max_rows_to_check):
        try:
            df_test = pd.read_excel(file_path, header=row_idx, nrows=5)

            if df_test.empty or len(df_test.columns) < 3:
                continue

            valid_columns = []
            score = 0

            for col in df_test.columns:
                if pd.notna(col) and isinstance(col, str):
                    col_clean = col.strip().lower()

                    if len(col_clean) > 0:
                        valid_columns.append(col)
                        score += 1

                        for keyword in drilling_keywords:
                            if keyword in col_clean:
                                score += 5
                                break

            # Check if data below header is numeric
            numeric_count = 0
            for col in valid_columns[:5]:
                try:
                    test_val = df_test[col].iloc[0]
                    if pd.notna(test_val):
                        float(str(test_val).replace(',', '').replace('-', '0'))
                        numeric_count += 1
                except:
                    pass

            if numeric_count >= 2:
                score += numeric_count * 2

            print(f"  Row {row_idx:2d}: {len(valid_columns):2d} valid cols, score={score:3d} | Sample: {valid_columns[:3]}")

            if score > max_score and len(valid_columns) >= 3:
                max_score = score
                best_header_row = row_idx

        except Exception as e:
            continue

    if best_header_row is None:
        print(f"  ⚠️  Could not detect header automatically. Using row 0.")
        best_header_row = 0
    else:
        print(f"  ✅ Best header row: {best_header_row} (score: {max_score})")

    return best_header_row


def standardize_column_names(df):
    """
    Standardize column names to match expected format
    ⭐ Also handles duplicate columns ⭐
    """
    column_mappings = {
        # Depth variations
        'depth': 'Depth', 'dept': 'Depth', 'md': 'Depth',
        'measured depth': 'Depth', 'measured_depth': 'Depth',

        # WOB variations
        'wob': 'WOB', 'mob': 'WOB', 'weight on bit': 'WOB',
        'weight_on_bit': 'WOB', 'weight': 'WOB',

        # ROP variations
        'rop': 'ROP', 'rate of penetration': 'ROP',
        'rate_of_penetration': 'ROP', 'penetration rate': 'ROP',
        'penetration_rate': 'ROP',

        # RPM variations
        'rpm': 'RPM', 'rotation': 'RPM', 'rotary speed': 'RPM',
        'rotary_speed': 'RPM', 'rotary': 'RPM',

        # Torque variations
        'torque': 'Surface_Torque', 'surface torque': 'Surface_Torque',
        'surface_torque': 'Surface_Torque', 'tq': 'Surface_Torque',
        'tor': 'Surface_Torque',

        # Flow rate variations
        'gpm': 'Q', 'q': 'Q', 'flow rate': 'Q', 'flow': 'Q',
        'flow_rate': 'Q', 'pump rate': 'Q', 'pump_rate': 'Q',

        # Pressure variations
        'spp': 'SPP', 'standpipe pressure': 'SPP',
        'standpipe_pressure': 'SPP', 'pressure': 'SPP',
        'pump pressure': 'SPP', 'pump_pressure': 'SPP',

        # Hook Load / Mud Weight variations
        'mw': 'Hook_Load', 'mud weight': 'Hook_Load',
        'mud_weight': 'Hook_Load', 'hook load': 'Hook_Load',
        'hookload': 'Hook_Load', 'hook_load': 'Hook_Load',
        'wt': 'Hook_Load',

        # Viscosity variations
        'vis': 'Viscosity', 'viscosity': 'Viscosity', 'visc': 'Viscosity',

        # Other common columns
        'time': 'Time', 'date': 'Date', 'hob': 'HOB',
        'bit depth': 'Depth', 'bit_depth': 'Depth',
    }

    # Create new column names
    new_columns = {}
    for col in df.columns:
        col_clean = str(col).lower().strip()
        col_clean = col_clean.replace('(', '').replace(')', '').replace('.', '').replace(':', '')
        col_clean = ' '.join(col_clean.split())

        if col_clean in column_mappings:
            new_columns[col] = column_mappings[col_clean]
        else:
            new_columns[col] = col

    df_renamed = df.rename(columns=new_columns)

    # Handle duplicate columns
    if df_renamed.columns.duplicated().any():
        print(f"\n⚠️  Warning: Duplicate columns detected after standardization!")
        duplicates = df_renamed.columns[df_renamed.columns.duplicated()].unique()
        print(f"  Duplicates: {list(duplicates)}")
        df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]
        print(f"  → Kept first occurrence of each duplicate column")

    return df_renamed


def load_drilling_data(file_path, columns_to_drop=None, verbose=True):
    """
    🚀 Universal data loader with automatic header detection
    
    Args:
        file_path: Path to Excel file
        columns_to_drop: List of column names to drop before processing
        verbose: Print detailed information
    
    Returns:
        DataFrame with cleaned numeric data
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"📂 LOADING: {os.path.basename(file_path)}")
        print(f"{'='*80}")
    
    # Step 1: Detect header row
    header_row = detect_header_row(file_path)
    
    # Step 2: Load data with detected header
    df = pd.read_excel(file_path, header=header_row)
    
    if verbose:
        print(f"\n📊 Initial data shape: {df.shape}")
        print(f"Columns found: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Step 3: Standardize column names
    df = standardize_column_names(df)
    
    if verbose:
        print(f"\n✅ Standardized columns: {list(df.columns)}")
    
    # Step 4: Drop unwanted columns
    if columns_to_drop:
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            if verbose:
                print(f"\n🗑️  Dropped columns: {existing_cols_to_drop}")
    
    # Step 5: Clean numeric columns
    if verbose:
        print(f"\n🔧 Converting to numeric...")
    
    for col in df.columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Step 6: Drop rows with NaN in critical columns
    critical_columns = ['WOB', 'RPM', 'Surface_Torque', 'Q']
    available_critical = [col for col in critical_columns if col in df.columns]
    
    if available_critical:
        initial_rows = len(df)
        df = df.dropna(subset=available_critical)
        dropped_rows = initial_rows - len(df)
        
        if verbose and dropped_rows > 0:
            print(f"\n🧹 Dropped {dropped_rows} rows with NaN in critical columns: {available_critical}")
    
    if verbose:
        print(f"\n✅ Final data shape: {df.shape}")
        print(f"{'='*80}\n")
    
    return df


def load_multiple_excel_files(file_paths, columns_to_drop=None, verbose=True):
    """
    Load and combine multiple Excel files
    
    Args:
        file_paths: List of file paths
        columns_to_drop: Columns to drop from each file
        verbose: Print info
    
    Returns:
        Combined DataFrame
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"📚 LOADING MULTIPLE TRAINING FILES")
        print(f"{'='*80}")
        print(f"Number of files: {len(file_paths)}")
    
    all_dataframes = []
    
    for file_path in file_paths:
        df = load_drilling_data(file_path, columns_to_drop=columns_to_drop, verbose=verbose)
        all_dataframes.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ COMBINED TRAINING DATA")
        print(f"{'='*80}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"{'='*80}\n")
    
    return combined_df


def load_training_and_test_data(train_files, test_file, columns_to_drop=None, verbose=True):
    """
    Load training data (from multiple files) and test data (from single file)
    
    Args:
        train_files: List of training file paths
        test_file: Single test file path
        columns_to_drop: Columns to drop
        verbose: Print info
    
    Returns:
        train_data, test_data (both as DataFrames)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"LOADING TRAINING AND TEST DATA FROM SEPARATE FILES")
        print(f"{'='*80}")
    
    # Load training data (possibly from multiple files)
    if isinstance(train_files, list):
        train_data = load_multiple_excel_files(train_files, columns_to_drop=columns_to_drop, verbose=verbose)
    else:
        train_data = load_drilling_data(train_files, columns_to_drop=columns_to_drop, verbose=verbose)
    
    # Load test data (single file)
    test_data = load_drilling_data(test_file, columns_to_drop=columns_to_drop, verbose=verbose)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Training data: {train_data.shape[0]} rows × {train_data.shape[1]} columns")
        print(f"Test data:     {test_data.shape[0]} rows × {test_data.shape[1]} columns")
        print(f"{'='*80}\n")
    
    return train_data, test_data


def prepare_train_test_datasets(train_data, test_data, target_column='ROP', 
                                 augment_train=True, noise_levels=[0.10]):
    """
    Prepare training and test datasets for ROP prediction
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame (from separate file)
        target_column: Target column name
        augment_train: Whether to augment training data
        noise_levels: Noise levels for augmentation
    
    Returns:
        Dictionary with all datasets
    """
    feature_columns = ['WOB', 'RPM', 'Surface_Torque', 'Q']
    
    # Extract features and targets
    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values
    
    X_test = test_data[feature_columns].values
    y_test = test_data[target_column].values
    
    print(f"\n{'='*80}")
    print(f"PREPARING DATASETS FOR {target_column} PREDICTION")
    print(f"{'='*80}")
    print(f"Original training set: {X_train.shape[0]} samples")
    print(f"Test set (from separate file): {X_test.shape[0]} samples")
    
    # Augment training data if requested
    if augment_train:
        X_train_aug, y_train_aug = augment_training_data(
            X_train, y_train,
            noise_levels=noise_levels,
            verbose=True
        )
    else:
        X_train_aug = X_train
        y_train_aug = y_train
    
    return {
        'X_train_original': X_train,
        'y_train_original': y_train,
        'X_train_augmented': X_train_aug,
        'y_train_augmented': y_train_aug,
        'X_test': X_test,
        'y_test': y_test
    }


def plot_rop_results(y_train, y_train_pred, y_test, y_test_pred, 
                     train_results, test_results):
    """
    Plot ROP prediction results (train vs test)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual ROP (m/hr)', fontsize=12)
    axes[0].set_ylabel('Predicted ROP (m/hr)', fontsize=12)
    axes[0].set_title(f'ROP - Training Set\nR² = {train_results["r2"]:.4f}, RMSE = {train_results["rmse"]:.4f}',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual ROP (m/hr)', fontsize=12)
    axes[1].set_ylabel('Predicted ROP (m/hr)', fontsize=12)
    axes[1].set_title(f'ROP - Test Set (Separate File)\nR² = {test_results["r2"]:.4f}, RMSE = {test_results["rmse"]:.4f}',
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rop_results_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved as 'rop_results_comparison.png'")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function with separate train/test files and hyperparameter optimization
    """
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: DEFINE FILE PATHS
    # ═══════════════════════════════════════════════════════════
    
    # Training files (can be multiple Excel files)
    train_files = [
        r'E:\Data\preprocessed\drill operation\human edit\Bit Data#1214#RR#34.xlsx',
        r'E:\Data\preprocessed\drill operation\human edit\Bit Data#1214#AZ#546.xlsx',  # Uncomment if you have more files
        # 'data/train_well3.xlsx',
    ]
    
    # Test file (single Excel file)
    test_file = r'E:\Data\preprocessed\drill operation\human edit\Bit Data#1214#MI#131.xlsx'
    
    # Columns to drop (optional)
    columns_to_drop = ['Time', 'Date', 'Operator', 'Comments', 'Lit']
    
    # ═══════════════════════════════════════════════════════════
    # STEP 2: LOAD TRAINING AND TEST DATA
    # ═══════════════════════════════════════════════════════════
    
    train_data, test_data = load_training_and_test_data(
        train_files=train_files,
        test_file=test_file,
        columns_to_drop=columns_to_drop,
        verbose=True
    )
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: OPTIMIZE HYPERPARAMETERS USING DE
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"🔧 OPTIMIZING ROP MODEL HYPERPARAMETERS USING DIFFERENTIAL EVOLUTION")
    print(f"{'='*80}")
    print(f"⏰ This may take several minutes...")
    print(f"{'='*80}\n")
    
    # Run DE optimization on TRAINING DATA ONLY
    best_params = optimize_ann_hyperparameters(
        data=train_data,  # ⭐ Only training data
        target_column='ROP',
        pop_size=10,      # Population size
        max_iter=20       # Generations
    )
    
    # Decode optimized parameters
    valid_lrs = [0.01, 0.001]
    lr_index = int(np.clip(best_params[2], 0, 2))
    
    optimized_config = {
        'hidden_neurons': int(best_params[0]),
        'dropout_rate': best_params[1],
        'learning_rate': valid_lrs[lr_index],
        'batch_size': int(best_params[3]),
        'weight_decay': best_params[4]
    }
    
    print(f"\n{'='*80}")
    print(f"✅ OPTIMIZED HYPERPARAMETERS")
    print(f"{'='*80}")
    for key, value in optimized_config.items():
        print(f"  {key:20s}: {value}")
    print(f"{'='*80}\n")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: PREPARE DATASETS FOR ROP PREDICTION
    # ═══════════════════════════════════════════════════════════
    
    rop_datasets = prepare_train_test_datasets(
        train_data=train_data,
        test_data=test_data,
        target_column='ROP',
        augment_train=True,
        noise_levels=[0.10]
    )
    
    # ═══════════════════════════════════════════════════════════
    # STEP 5: TRAIN ROP MODEL WITH OPTIMIZED HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING ROP MODEL WITH OPTIMIZED HYPERPARAMETERS")
    print(f"{'='*80}")
    
    rop_model = DrillingPredictor(
        hidden_sizes=[optimized_config['hidden_neurons']],  # ⭐ Optimized
        learning_rate=optimized_config['learning_rate'],    # ⭐ Optimized
        batch_size=optimized_config['batch_size'],          # ⭐ Optimized
        epochs=200,                                         # Fixed
        dropout_rate=optimized_config['dropout_rate'],      # ⭐ Optimized
        weight_decay=optimized_config['weight_decay']       # ⭐ Optimized
    )
    
    # Create validation split from augmented training data
    X_train_aug = rop_datasets['X_train_augmented']
    y_train_aug = rop_datasets['y_train_augmented']
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_aug, y_train_aug, test_size=0.2, random_state=42
    )
    
    # Train model
    rop_model.train(X_train_split, y_train_split, X_val, y_val)
    
    # ═══════════════════════════════════════════════════════════
    # STEP 6: EVALUATE ON SEPARATE TEST FILE
    # ═══════════════════════════════════════════════════════════
    
    X_test = rop_datasets['X_test']
    y_test = rop_datasets['y_test']
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATING ON SEPARATE TEST FILE")
    print(f"{'='*80}")
    
    # Evaluate on test set
    rop_results_test = rop_model.evaluate(X_test, y_test)
    
    print(f"\n✅ ROP Model - TEST SET Performance (from {os.path.basename(test_file)}):")
    print(f"  RMSE: {rop_results_test['rmse']:.4f}")
    print(f"  R²: {rop_results_test['r2']:.4f}")
    print(f"  AARE: {rop_results_test['aare']:.2f}%")
    
    # Evaluate on original training set
    X_train_orig = rop_datasets['X_train_original']
    y_train_orig = rop_datasets['y_train_original']
    
    rop_results_train = rop_model.evaluate(X_train_orig, y_train_orig)
    
    print(f"\n✅ ROP Model - TRAIN SET Performance (from training files):")
    print(f"  RMSE: {rop_results_train['rmse']:.4f}")
    print(f"  R²: {rop_results_train['r2']:.4f}")
    print(f"  AARE: {rop_results_train['aare']:.2f}%")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 7: SAVE RESULTS
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"💾 SAVING RESULTS")
    print(f"{'='*80}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'Actual_ROP': y_test,
        'Predicted_ROP': rop_results_test['predictions'],
        'Error': y_test - rop_results_test['predictions'],
        'Absolute_Error': np.abs(y_test - rop_results_test['predictions']),
        'Relative_Error_%': np.abs((y_test - rop_results_test['predictions']) / y_test) * 100
    })
    
    results_df.to_csv('rop_predictions_on_test_file.csv', index=False)
    print(f"  ✅ Predictions saved to 'rop_predictions_on_test_file.csv'")
    
    # Save final report
    final_report = pd.DataFrame({
        'Metric': ['RMSE', 'AARE (%)', 'R²'],
        'ROP (Train)': [
            rop_results_train['rmse'],
            rop_results_train['aare'],
            rop_results_train['r2']
        ],
        'ROP (Test)': [
            rop_results_test['rmse'],
            rop_results_test['aare'],
            rop_results_test['r2']
        ]
    })
    
    final_report.to_csv('final_report_rop.csv', index=False)
    print(f"  ✅ Final report saved to 'final_report_rop.csv'")
    
    # Save optimized hyperparameters
    hyperparams_df = pd.DataFrame({
        'Parameter': list(optimized_config.keys()),
        'Optimized_Value': list(optimized_config.values())
    })
    
    hyperparams_df.to_csv('optimized_hyperparameters.csv', index=False)
    print(f"  ✅ Hyperparameters saved to 'optimized_hyperparameters.csv'")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 8: PLOT RESULTS
    # ═══════════════════════════════════════════════════════════
    
    plot_rop_results(
        y_train_orig, rop_results_train['predictions'],
        y_test, rop_results_test['predictions'],
        rop_results_train, rop_results_test
    )
    
    print(f"\n{'='*80}")
    print(f"✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    return rop_model, rop_datasets, rop_results_train, rop_results_test, optimized_config


if __name__ == "__main__":
    model, datasets, train_results, test_results, hyperparams = main()
