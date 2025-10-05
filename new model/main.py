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
from scipy.optimize import differential_evolution

# Import custom modules
from drilling_xgboost import train_drilling_models, DrillingXGBoostPredictor
from xgboost_hyperparameter_optimizer import optimize_xgboost_hyperparameters
from baseline_models import train_baseline_models
from de_optimizer import DifferentialEvolution

def clean_numeric_column(series):
    """
    Clean a pandas series containing mixed numeric and string data
    ⭐ GUARANTEED to return float64 dtype or raise clear error ⭐
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

    # Double-check: convert to numeric (this catches any remaining strings)
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
    🔍 اتوماتیک پیدا می‌کنه که header از کدوم سطر شروع میشه

    بررسی می‌کنه:
    - اولین سطری که حداقل 3 ستون با نام معنی‌دار داره
    - نام ستون‌ها باید string باشن (نه عدد، نه NaN)
    - کلمات کلیدی drilling رو داشته باشه

    Returns:
        header_row (int): شماره سطری که header شروع میشه
    """
    print(f"\n🔍 Detecting header row...")

    # کلمات کلیدی که معمولاً تو header هستن
    drilling_keywords = [
        'depth', 'wob', 'mob', 'rop', 'rpm', 'torque', 'gpm', 'spp',
        'mw', 'vis', 'flow', 'pressure', 'rate', 'bit', 'hole',
        'md', 'tvd', 'inc', 'azi', 'dls'  # Directional drilling terms
    ]

    best_header_row = None
    max_score = 0

    for row_idx in range(max_rows_to_check):
        try:
            # سعی می‌کنیم این سطر رو به عنوان header بخونیم
            df_test = pd.read_excel(file_path, header=row_idx, nrows=5)

            if df_test.empty or len(df_test.columns) < 3:
                continue

            # شمارش ستون‌های معتبر
            valid_columns = []
            score = 0

            for col in df_test.columns:
                # باید string باشه و خالی نباشه
                if pd.notna(col) and isinstance(col, str):
                    col_clean = col.strip().lower()

                    if len(col_clean) > 0:
                        valid_columns.append(col)
                        score += 1

                        # امتیاز بیشتر اگه کلمه کلیدی داشته باشه
                        for keyword in drilling_keywords:
                            if keyword in col_clean:
                                score += 5
                                break

            # بررسی کنیم که آیا داده‌های زیرش numeric هستن؟
            numeric_count = 0
            for col in valid_columns[:5]:  # فقط 5 تا اول
                try:
                    # اگه بتونیم به عدد تبدیل کنیم
                    test_val = df_test[col].iloc[0]
                    if pd.notna(test_val):
                        float(str(test_val).replace(',', '').replace('-', '0'))
                        numeric_count += 1
                except:
                    pass

            # اگه زیر header عدد بود، امتیاز بیشتر
            if numeric_count >= 2:
                score += numeric_count * 2

            print(f"  Row {row_idx:2d}: {len(valid_columns):2d} valid cols, score={score:3d} | Sample: {valid_columns[:3]}")

            # بهترین رو نگه دار
            if score > max_score and len(valid_columns) >= 3:
                max_score = score
                best_header_row = row_idx

        except Exception as e:
            # اگه نتونستیم بخونیم، رد کن
            continue

    if best_header_row is None:
        print(f"  ⚠️  Could not detect header automatically. Using row 0.")
        best_header_row = 0
    else:
        print(f"  ✅ Best header row: {best_header_row} (score: {max_score})")

    return best_header_row

def convert_range_to_number(value):
    """
    Convert range strings like '15-25' to their AVERAGE (20.0)
    Handle various formats including single numbers, ranges, and non-numeric values

    ⭐ ALWAYS returns float or np.nan (NEVER string) ⭐
    """
    # ⭐ FIX: Handle DataFrame (happens with duplicate columns) ⭐
    if isinstance(value, pd.DataFrame):
        # Take the first column if it's a DataFrame
        if len(value.columns) > 0:
            value = value.iloc[:, 0]
        else:
            return np.nan

    # ⭐ FIX: Handle Series (shouldn't happen in .apply(), but just in case) ⭐
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

    # ⭐ Pattern for ranges: "20-25", "20 - 25", "20to25", etc. ⭐
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
                # ⭐ محاسبه میانگین ⭐
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
    print(f"⚠️  Warning: Could not convert '{value}' to number. Returning NaN.")
    return np.nan

def standardize_column_names(df):
    """
    Standardize column names to match expected format
    Handles different naming conventions
    ⭐ FIX: Also handles duplicate columns by keeping only the first occurrence ⭐
    """
    # Comprehensive mapping from various possible names to standard names
    column_mappings = {
        # Depth variations
        'depth': 'Depth',
        'dept': 'Depth',
        'md': 'Depth',
        'measured depth': 'Depth',
        'measured_depth': 'Depth',

        # WOB variations (both 'mob' and 'wob' map to 'WOB')
        'wob': 'WOB',
        'mob': 'WOB',  # ⭐ This causes duplicate 'WOB' columns ⭐
        'weight on bit': 'WOB',
        'weight_on_bit': 'WOB',
        'weight': 'WOB',

        # ROP variations
        'rop': 'ROP',
        'rate of penetration': 'ROP',
        'rate_of_penetration': 'ROP',
        'penetration rate': 'ROP',
        'penetration_rate': 'ROP',

        # RPM variations
        'rpm': 'RPM',
        'rotation': 'RPM',
        'rotary speed': 'RPM',
        'rotary_speed': 'RPM',
        'rotary': 'RPM',

        # Torque variations
        'torque': 'Surface_Torque',
        'surface torque': 'Surface_Torque',
        'surface_torque': 'Surface_Torque',
        'tq': 'Surface_Torque',
        'tor': 'Surface_Torque',

        # Flow rate variations
        'gpm': 'Q',
        'q': 'Q',
        'flow rate': 'Q',
        'flow': 'Q',
        'flow_rate': 'Q',
        'pump rate': 'Q',
        'pump_rate': 'Q',

        # Pressure variations
        'spp': 'SPP',
        'standpipe pressure': 'SPP',
        'standpipe_pressure': 'SPP',
        'pressure': 'SPP',
        'pump pressure': 'SPP',
        'pump_pressure': 'SPP',

        # Hook Load / Mud Weight variations
        'mw': 'Hook_Load',
        'mud weight': 'Hook_Load',
        'mud_weight': 'Hook_Load',
        'hook load': 'Hook_Load',
        'hookload': 'Hook_Load',
        'hook_load': 'Hook_Load',
        'wt': 'Hook_Load',

        # Viscosity variations
        'vis': 'Viscosity',
        'viscosity': 'Viscosity',
        'visc': 'Viscosity',

        # Other common columns
        'time': 'Time',
        'date': 'Date',
        'hob': 'HOB',
        'bit depth': 'Depth',
        'bit_depth': 'Depth',
    }

    # Create new column names
    new_columns = {}
    for col in df.columns:
        # Clean the column name
        col_clean = str(col).lower().strip()
        col_clean = col_clean.replace('(', '').replace(')', '').replace('.', '').replace(':', '')
        col_clean = ' '.join(col_clean.split())  # Remove extra spaces

        if col_clean in column_mappings:
            new_columns[col] = column_mappings[col_clean]
        else:
            # Keep original if not in mapping
            new_columns[col] = col

    df_renamed = df.rename(columns=new_columns)

    # ⭐ FIX: Handle duplicate columns by keeping only the first occurrence ⭐
    # This happens when both 'Mob' and 'Wob' exist and both map to 'WOB'
    if df_renamed.columns.duplicated().any():
        print(f"\n⚠️  Warning: Duplicate columns detected after standardization!")
        duplicates = df_renamed.columns[df_renamed.columns.duplicated()].unique()
        print(f"  Duplicates: {list(duplicates)}")

        # Keep only the first occurrence of each column
        df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]
        print(f"  → Kept first occurrence of each duplicate column")

    return df_renamed

def load_drilling_data(file_path, columns_to_drop=None, verbose=True):
    """
    🚀 Universal data loader که به صورت اتوماتیک header رو پیدا می‌کنه
    و همه string ها رو به float تبدیل می‌کنه

    Args:
        file_path: Path to Excel file
        columns_to_drop: List of column names to drop before processing
        verbose: Print detailed information

    Returns:
        Clean pandas DataFrame with standardized columns (ALL NUMERIC)
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    if verbose:
        print(f"\n{'='*80}")
        print(f"📂 Loading: {os.path.basename(file_path)}")
        print(f"{'='*80}")

    try:
        # ⭐ Step 1: پیدا کردن اتوماتیک header row ⭐
        header_row = detect_header_row(file_path, max_rows_to_check=15)

        # ⭐ Step 2: بارگذاری داده با header درست ⭐
        data = pd.read_excel(file_path, header=header_row)

        if verbose:
            print(f"\n✓ Loaded {data.shape[0]} rows × {data.shape[1]} columns")
            print(f"  Raw columns: {list(data.columns)[:10]}")

        # Step 3: Drop specified columns (before standardization)
        if columns_to_drop is not None:
            if not isinstance(columns_to_drop, list):
                columns_to_drop = [columns_to_drop]

            existing_cols = [col for col in columns_to_drop if col in data.columns]
            if existing_cols:
                data = data.drop(columns=existing_cols)
                if verbose:
                    print(f"\n🗑️  Dropped columns: {existing_cols}")

        # Step 4: Standardize column names
        data = standardize_column_names(data)

        if verbose:
            print(f"\n✓ Standardized columns:")
            for i, col in enumerate(data.columns, 1):
                print(f"  {i:2d}. {col}")

        # Step 5: Define required and optional columns
        required_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'ROP', 'Surface_Torque']
        optional_columns = ['Hook_Load', 'Viscosity', 'Time', 'Date', 'HOB']

        # Step 6: Check availability
        available_required = [col for col in required_columns if col in data.columns]
        available_optional = [col for col in optional_columns if col in data.columns]
        missing_required = [col for col in required_columns if col not in data.columns]

        if verbose:
            print(f"\n📊 Column Status:")
            print(f"  ✓ Available required ({len(available_required)}/{len(required_columns)}): {available_required}")
            if available_optional:
                print(f"  ✓ Available optional ({len(available_optional)}): {available_optional}")
            if missing_required:
                print(f"  ⚠️  Missing required ({len(missing_required)}): {missing_required}")

        # Step 7: Handle Hook_Load substitution
        if 'Hook_Load' not in data.columns:
            if 'Viscosity' in data.columns:
                data['Hook_Load'] = data['Viscosity'].copy()
                if 'Hook_Load' not in available_required:
                    available_required.append('Hook_Load')
                if verbose:
                    print(f"  ℹ️  Using 'Viscosity' as Hook_Load substitute")
            elif 'Mw' in data.columns:
                data['Hook_Load'] = data['Mw'].copy()
                if 'Hook_Load' not in available_required:
                    available_required.append('Hook_Load')
                if verbose:
                    print(f"  ℹ️  Using 'Mw' as Hook_Load substitute")

        # Step 8: Clean all numeric columns
        all_columns_to_clean = list(set(available_required + available_optional))

        if verbose:
            print(f"\n🧹 Cleaning {len(all_columns_to_clean)} numeric columns...")
            print(f"{'='*80}")

        for col in all_columns_to_clean:
            if col in data.columns:
                try:
                    # Store original for analysis
                    original_series = data[col].copy()

                    # ⭐ اینجا اتفاق می‌افته: تبدیل string به float ⭐
                    data[col] = clean_numeric_column(original_series)

                    # Verify it's numeric now
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(f"❌ Column '{col}' is still not numeric! Dtype: {data[col].dtype}")

                    if verbose:
                        valid_count = data[col].notna().sum()
                        total_count = len(data)
                        valid_pct = 100 * valid_count / total_count

                        if valid_count > 0:
                            col_min = data[col].min()
                            col_max = data[col].max()
                            col_mean = data[col].mean()

                            # بررسی چند تا رنج تبدیل شده
                            range_count = 0
                            for orig_val in original_series.head(100):
                                val_str = str(orig_val)
                                if '-' in val_str and val_str.count('-') == 1:
                                    # Make sure it's a range, not negative number
                                    if re.search(r'\d+\s*-\s*\d+', val_str):
                                        range_count += 1

                            range_info = f" | Ranges: {range_count}" if range_count > 0 else ""

                            print(f"  ✓ {col:15s}: {valid_count:4d}/{total_count} ({valid_pct:5.1f}%) | "
                                  f"[{col_min:10.2f}, {col_max:10.2f}] | μ={col_mean:10.2f}{range_info}")
                        else:
                            print(f"  ⚠️  {col:15s}: {valid_count:4d}/{total_count} ({valid_pct:5.1f}%) | All NaN!")

                except Exception as e:
                    if verbose:
                        print(f"  ❌ {col:15s}: FAILED - {str(e)[:60]}")
                    raise  # Re-raise to stop execution

        print(f"{'='*80}")

        # Step 9: Select final columns (only those available)
        final_columns = [col for col in required_columns if col in data.columns]
        if 'Hook_Load' in data.columns and 'Hook_Load' not in final_columns:
            final_columns.append('Hook_Load')

        # Check if we have minimum required columns
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
    ⭐ NEW: Train models using separate training and testing datasets ⭐
    
    Args:
        train_data: DataFrame with training data
        test_data: DataFrame with testing data
        optimize_hyperparams: Whether to optimize XGBoost hyperparameters
    
    Returns:
        Dictionary containing trained models and results
    """
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    
    # Verify both datasets have required columns
    required_cols = feature_columns + ['ROP', 'Surface_Torque']
    for dataset_name, dataset in [('Training', train_data), ('Testing', test_data)]:
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"{dataset_name} data missing columns: {missing_cols}")
    
    # Prepare training data
    X_train = train_data[feature_columns].values
    y_rop_train = train_data['ROP'].values
    y_torque_train = train_data['Surface_Torque'].values
    
    # Prepare testing data
    X_test = test_data[feature_columns].values
    y_rop_test = test_data['ROP'].values
    y_torque_test = test_data['Surface_Torque'].values
    
    print(f"\n{'='*80}")
    print(f"📊 Dataset Information:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set:  {len(X_test)} samples")
    print(f"{'='*80}")
    
    results = {}
    
    # ====== STEP 1: Train Baseline Models ======
    print(f"\n{'='*80}")
    print("STEP 1: Training Baseline Models (Ridge & Random Forest)")
    print(f"{'='*80}")
    
    # Combine train and test for baseline (they expect to do their own split)
    # Or we can modify baseline_models.py to accept separate train/test
    # For now, let's just use train data and evaluate on test
    baseline_results = train_baseline_models(train_data)
    results['baseline'] = baseline_results
    
    # ====== STEP 2: Hyperparameter Optimization (Optional) ======
    best_params_rop = None
    best_params_torque = None
    
    if optimize_hyperparams:
        print(f"\n{'='*80}")
        print("STEP 2: XGBoost Hyperparameter Optimization")
        print(f"{'='*80}")
        
        # Split training data for validation during optimization
        X_train_opt, X_val_opt, y_rop_train_opt, y_rop_val_opt = train_test_split(
            X_train, y_rop_train, test_size=0.2, random_state=42
        )
        _, _, y_torque_train_opt, y_torque_val_opt = train_test_split(
            X_train, y_torque_train, test_size=0.2, random_state=42
        )
        
        # Optimize ROP model
        print("\n🎯 Optimizing ROP Model Hyperparameters...")
        best_params_rop, rop_opt_history = optimize_xgboost_hyperparameters(
            X_train_opt, y_rop_train_opt,
            X_val_opt, y_rop_val_opt,
            target_name='ROP',
            pop_size=20,
            max_iter=30
        )
        
        # Optimize Torque model
        print("\n🎯 Optimizing Torque Model Hyperparameters...")
        best_params_torque, torque_opt_history = optimize_xgboost_hyperparameters(
            X_train_opt, y_torque_train_opt,
            X_val_opt, y_torque_val_opt,
            target_name='Torque',
            pop_size=20,
            max_iter=30
        )
        
        results['hyperparameter_optimization'] = {
            'rop_params': best_params_rop,
            'torque_params': best_params_torque,
            'rop_history': rop_opt_history,
            'torque_history': torque_opt_history
        }
    
    # ====== STEP 3: Train XGBoost Models ======
    print(f"\n{'='*80}")
    print("STEP 3: Training XGBoost Models")
    print(f"{'='*80}")
    
    # Train ROP model
    print("\n🎯 Training ROP Model...")
    rop_model = DrillingXGBoostPredictor(
        n_estimators=int(best_params_rop['n_estimators']) if best_params_rop else 100,
        max_depth=int(best_params_rop['max_depth']) if best_params_rop else 6,
        learning_rate=best_params_rop['learning_rate'] if best_params_rop else 0.1,
        subsample=best_params_rop['subsample'] if best_params_rop else 0.8,
        colsample_bytree=best_params_rop['colsample_bytree'] if best_params_rop else 0.8,
        gamma=best_params_rop['gamma'] if best_params_rop else 0,
        min_child_weight=best_params_rop['min_child_weight'] if best_params_rop else 1,
        reg_alpha=best_params_rop['reg_alpha'] if best_params_rop else 0,
        reg_lambda=best_params_rop['reg_lambda'] if best_params_rop else 1
    )
    
    # For early stopping, split training data into train/val
    X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
        X_train, y_rop_train, test_size=0.2, random_state=42
    )
    
    rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
    rop_results = rop_model.evaluate(X_test, y_rop_test)
    
    print(f"\n✓ ROP Model Results:")
    print(f"  RMSE: {rop_results['rmse']:.4f}")
    print(f"  R²: {rop_results['r2']:.4f}")
    print(f"  AARE: {rop_results['aare']:.2f}%")
    
    # Train Torque model
    print("\n🎯 Training Torque Model...")
    torque_model = DrillingXGBoostPredictor(
        n_estimators=int(best_params_torque['n_estimators']) if best_params_torque else 100,
        max_depth=int(best_params_torque['max_depth']) if best_params_torque else 6,
        learning_rate=best_params_torque['learning_rate'] if best_params_torque else 0.1,
        subsample=best_params_torque['subsample'] if best_params_torque else 0.8,
        colsample_bytree=best_params_torque['colsample_bytree'] if best_params_torque else 0.8,
        gamma=best_params_torque['gamma'] if best_params_torque else 0,
        min_child_weight=best_params_torque['min_child_weight'] if best_params_torque else 1,
        reg_alpha=best_params_torque['reg_alpha'] if best_params_torque else 0,
        reg_lambda=best_params_torque['reg_lambda'] if best_params_torque else 1
    )
    
    _, _, y_torque_train_split, y_torque_val_split = train_test_split(
        X_train, y_torque_train, test_size=0.2, random_state=42
    )
    
    torque_model.train(X_train_split, y_torque_train_split, X_val_split, y_torque_val_split)
    torque_results = torque_model.evaluate(X_test, y_torque_test)
    
    print(f"\n✓ Torque Model Results:")
    print(f"  RMSE: {torque_results['rmse']:.4f}")
    print(f"  R²: {torque_results['r2']:.4f}")
    print(f"  AARE: {torque_results['aare']:.2f}%")
    
    results['xgboost'] = {
        'rop_model': rop_model,
        'torque_model': torque_model,
        'rop_results': rop_results,
        'torque_results': torque_results
    }
    
    return results


def generate_report(results, output_dir='results'):
    """
    Generate comprehensive report with visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*80}")
    print("📊 GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")
    
    # ====== Model Comparison Plot ======
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    models = ['Ridge', 'Random Forest', 'XGBoost']
    
    # ROP R² scores
    rop_r2_scores = [
        results['baseline']['rop_ridge_r2'],
        results['baseline']['rop_rf_r2'],
        results['xgboost']['rop_results']['r2']
    ]
    
    # Torque R² scores
    torque_r2_scores = [
        results['baseline']['torque_ridge_r2'],
        results['baseline']['torque_rf_r2'],
        results['xgboost']['torque_results']['r2']
    ]
    
    # Plot ROP
    axes[0].bar(models, rop_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('ROP Prediction - Model Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(rop_r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Plot Torque
    axes[1].bar(models, torque_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Torque Prediction - Model Comparison')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(torque_r2_scores):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_comparison.png")
    plt.close()
    
    # ====== Optimization History (if available) ======
    if 'hyperparameter_optimization' in results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROP optimization history
        rop_history = results['hyperparameter_optimization']['rop_history']
        axes[0].plot(rop_history, 'b-', linewidth=2)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Negative R² (Fitness)')
        axes[0].set_title('ROP Hyperparameter Optimization')
        axes[0].grid(True, alpha=0.3)
        
        # Torque optimization history
        torque_history = results['hyperparameter_optimization']['torque_history']
        axes[1].plot(torque_history, 'r-', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Negative R² (Fitness)')
        axes[1].set_title('Torque Hyperparameter Optimization')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/hyperparameter_optimization.png")
        plt.close()
    
    # ====== Drilling Parameter Optimization Results (if available) ======
    if 'drilling_optimization' in results:
        opt_results = results['drilling_optimization']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(opt_results['fitness_history'], 'g-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Negative ROP + Penalty)')
        ax.set_title('Drilling Parameter Optimization Convergence')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drilling_optimization_convergence.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/drilling_optimization_convergence.png")
        plt.close()
        
        # Print optimization results
        print(f"\n🎯 Optimal Drilling Parameters:")
        print(f"  WOB: {opt_results['optimal_wob']:.2f}")
        print(f"  RPM: {opt_results['optimal_rpm']:.2f}")
        print(f"  Predicted ROP: {opt_results['predicted_rop']:.4f}")
        print(f"  Predicted Torque: {opt_results['predicted_torque']:.2f} Lb.Ft")
    
    print(f"\n✅ Report generation completed!")
    print(f"  All results saved to: {output_dir}/")


def main():
    """
    ⭐ UPDATED: Main execution flow with separate train/test files ⭐
    """
    print(f"\n{'='*80}")
    print("🚀 DRILLING OPTIMIZATION SYSTEM - XGBoost Version")
    print(f"{'='*80}")
    
    # ====== Configuration ======
    TRAIN_FILE = 'train_data.xlsx'  # ⭐ Training data file ⭐
    TEST_FILE = 'test_data.xlsx'    # ⭐ Testing data file ⭐
    OPTIMIZE_HYPERPARAMETERS = True  # Set to True to optimize XGBoost hyperparameters
    OPTIMIZE_DRILLING_PARAMS = True  # Set to True to optimize WOB/RPM
    
    # ====== Load Training Data ======
    print(f"\n{'='*80}")
    print("LOADING TRAINING DATA")
    print(f"{'='*80}")
    
    train_data = load_drilling_data(
        TRAIN_FILE,
        columns_to_drop=None,
        verbose=True
    )
    
    # ====== Load Testing Data ======
    print(f"\n{'='*80}")
    print("LOADING TESTING DATA")
    print(f"{'='*80}")
    
    test_data = load_drilling_data(
        TEST_FILE,
        columns_to_drop=None,
        verbose=True
    )
    
    # ====== Train Models ======
    results = train_models_with_separate_data(
        train_data,
        test_data,
        optimize_hyperparams=OPTIMIZE_HYPERPARAMETERS
    )
    
    # ====== Optimize Drilling Parameters (Optional) ======
    if OPTIMIZE_DRILLING_PARAMS:
        print(f"\n{'='*80}")
        print("STEP 4: Optimizing Drilling Parameters (WOB, RPM)")
        print(f"{'='*80}")
        
        # Use first row of test data as fixed parameters
        fixed_params = test_data[['SPP', 'Q', 'Depth', 'Hook_Load']].iloc[0].values
        
        print(f"\n📍 Fixed Parameters:")
        print(f"  SPP: {fixed_params[0]:.2f}")
        print(f"  Q: {fixed_params[1]:.2f}")
        print(f"  Depth: {fixed_params[2]:.2f}")
        print(f"  Hook_Load: {fixed_params[3]:.2f}")
        
        # Define bounds for WOB and RPM
        bounds = [
            (train_data['WOB'].min(), train_data['WOB'].max()),    # WOB bounds
            (train_data['RPM'].min(), train_data['RPM'].max())     # RPM bounds
        ]
        
        print(f"\n🎯 Optimization Bounds:")
        print(f"  WOB: [{bounds[0][0]:.2f}, {bounds[0][1]:.2f}]")
        print(f"  RPM: [{bounds[1][0]:.2f}, {bounds[1][1]:.2f}]")
        
        # Create wrapper class for optimization
        class ModelWrapper:
            def __init__(self, rop_model, torque_model):
                self.rop_model = rop_model
                self.torque_model = torque_model
            
            def predict_rop(self, params):
                return self.rop_model.predict(params.reshape(1, -1))[0]
            
            def predict_torque(self, params):
                return self.torque_model.predict(params.reshape(1, -1))[0]
        
        model_wrapper = ModelWrapper(
            results['xgboost']['rop_model'],
            results['xgboost']['torque_model']
        )
        
        # Run optimization
        de_optimizer = DifferentialEvolution(
            pop_size=30,
            F=0.5,
            CR=0.7,
            max_iter=50
        )
        
        opt_results = de_optimizer.optimize(
            model_wrapper,
            fixed_params,
            bounds,
            verbose=True
        )
        
        results['drilling_optimization'] = opt_results
    
    # ====== Generate Report ======
    generate_report(results, output_dir='results')
    
    print(f"\n{'='*80}")
    print("✅ ALL PROCESSES COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
