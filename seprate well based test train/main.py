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




def plot_prediction_comparison(y_train_actual, y_train_pred, 
                                y_test_actual, y_test_pred, 
                                target_name='ROP', 
                                save_path=None):
    """
    Plot prediction vs actual for both training and testing sets
    
    Args:
        y_train_actual: Actual training values
        y_train_pred: Predicted training values
        y_test_actual: Actual test values
        y_test_pred: Predicted test values
        target_name: Name of target variable ('ROP' or 'Torque')
        save_path: Path to save figure (optional)
    """
    # Calculate R² scores
    r2_train = r2_score(y_train_actual, y_train_pred)
    r2_test = r2_score(y_test_actual, y_test_pred)
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme matching RF plots
    train_color = 'steelblue' if target_name == 'ROP' else 'darkseagreen'
    test_color = 'darkorange' if target_name == 'ROP' else 'mediumpurple'
    
    # ===== Training Set Plot =====
    ax = axes[0]
    
    # Scatter plot
    ax.scatter(y_train_actual, y_train_pred, 
               alpha=0.6, s=50, color=train_color, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_train_actual.min(), y_train_pred.min())
    max_val = max(y_train_actual.max(), y_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # Formatting
    ax.set_xlabel(f'Actual {target_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'{target_name} - Training Set (R² = {r2_train:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== Testing Set Plot =====
    ax = axes[1]
    
    # Scatter plot
    ax.scatter(y_test_actual, y_test_pred, 
               alpha=0.6, s=50, color=test_color, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test_actual.min(), y_test_pred.min())
    max_val = max(y_test_actual.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # Formatting
    ax.set_xlabel(f'Actual {target_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'{target_name} - Testing Set (R² = {r2_test:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_both_models_comparison(rop_train_actual, rop_train_pred,
                                 rop_test_actual, rop_test_pred,
                                 torque_train_actual, torque_train_pred,
                                 torque_test_actual, torque_test_pred,
                                 save_path=None):
    """
    Plot all 4 subplots (ROP train/test + Torque train/test)
    
    Args:
        rop_*: ROP actual and predicted values
        torque_*: Torque actual and predicted values
        save_path: Path to save figure (optional)
    """
    # Calculate R² scores
    r2_rop_train = r2_score(rop_train_actual, rop_train_pred)
    r2_rop_test = r2_score(rop_test_actual, rop_test_pred)
    r2_torque_train = r2_score(torque_train_actual, torque_train_pred)
    r2_torque_test = r2_score(torque_test_actual, torque_test_pred)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # ===== ROP Training =====
    ax = axes[0, 0]
    ax.scatter(rop_train_actual, rop_train_pred, 
               alpha=0.6, s=50, color='steelblue', edgecolors='white', linewidth=0.5)
    min_val = min(rop_train_actual.min(), rop_train_pred.min())
    max_val = max(rop_train_actual.max(), rop_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual ROP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted ROP', fontsize=12, fontweight='bold')
    ax.set_title(f'ROP - Training Set (R² = {r2_rop_train:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== ROP Testing =====
    ax = axes[0, 1]
    ax.scatter(rop_test_actual, rop_test_pred, 
               alpha=0.6, s=50, color='darkorange', edgecolors='white', linewidth=0.5)
    min_val = min(rop_test_actual.min(), rop_test_pred.min())
    max_val = max(rop_test_actual.max(), rop_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual ROP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted ROP', fontsize=12, fontweight='bold')
    ax.set_title(f'ROP - Testing Set (R² = {r2_rop_test:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== Torque Training =====
    ax = axes[1, 0]
    ax.scatter(torque_train_actual, torque_train_pred, 
               alpha=0.6, s=50, color='darkseagreen', edgecolors='white', linewidth=0.5)
    min_val = min(torque_train_actual.min(), torque_train_pred.min())
    max_val = max(torque_train_actual.max(), torque_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Torque', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Torque', fontsize=12, fontweight='bold')
    ax.set_title(f'Torque - Training Set (R² = {r2_torque_train:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== Torque Testing =====
    ax = axes[1, 1]
    ax.scatter(torque_test_actual, torque_test_pred, 
               alpha=0.6, s=50, color='mediumpurple', edgecolors='white', linewidth=0.5)
    min_val = min(torque_test_actual.min(), torque_test_pred.min())
    max_val = max(torque_test_actual.max(), torque_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Torque', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Torque', fontsize=12, fontweight='bold')
    ax.set_title(f'Torque - Testing Set (R² = {r2_torque_test:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()
    
    return fig




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
        
        # ⭐ حذف سطرهایی که همه مقادیرشون NaN هست ⭐
        data_clean = data_clean.dropna(how='all')
        
        # ⭐ حذف سطرهایی که هر کدوم از ستون‌های کلیدی NaN باشه ⭐
        data_clean = data_clean.dropna(subset=final_columns)
        
        # ⭐ FINAL CHECK: مطمئن بشیم که همه چی numeric شده ⭐
        for col in data_clean.columns:
            if not pd.api.types.is_numeric_dtype(data_clean[col]):
                # Try one more time to convert
                print(f"\n⚠️  WARNING: Column '{col}' is not numeric (dtype: {data_clean[col].dtype})")
                print(f"  Sample values: {data_clean[col].head(5).tolist()}")
                print(f"  Attempting force conversion...")
                data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
                
                if not pd.api.types.is_numeric_dtype(data_clean[col]):
                    raise ValueError(f"❌ Cannot convert column '{col}' to numeric!")
        
        rows_removed = len(data) - len(data_clean)
        rows_removed_pct = 100 * rows_removed / len(data) if len(data) > 0 else 0
        
        if verbose:
            print(f"\n📈 Cleaning Summary:")
            print(f"  Original rows: {len(data)}")
            print(f"  Clean rows:    {len(data_clean)}")
            print(f"  Removed:       {rows_removed} ({rows_removed_pct:.1f}%)")
            print(f"  Final columns: {final_columns}")
        
        if len(data_clean) == 0:
            raise ValueError("❌ No valid data remaining after cleaning!")
        
        if verbose:
            print(f"\n📋 Sample of cleaned data (first 3 rows):")
            print(data_clean.head(3).to_string())
            
            print(f"\n📊 Data Types (MUST ALL BE float64):")
            print(data_clean.dtypes)
            
            print(f"\n📊 Statistical Summary:")
            print(data_clean.describe().round(2).to_string())
            
            print(f"\n{'='*80}\n")
        
        return data_clean
        
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        raise



def find_common_columns(data1, data2, verbose=True):
    """Find common columns between two datasets"""
    cols1 = set(data1.columns)
    cols2 = set(data2.columns)
    common = sorted(cols1.intersection(cols2))
    
    if verbose:
        print(f"\n🔍 Finding Common Columns:")
        print(f"  Dataset 1: {sorted(cols1)}")
        print(f"  Dataset 2: {sorted(cols2)}")
        print(f"  ✓ Common:  {common}")
        
        only_in_1 = sorted(cols1 - cols2)
        only_in_2 = sorted(cols2 - cols1)
        
        if only_in_1:
            print(f"  ⚠️  Only in dataset 1: {only_in_1}")
        if only_in_2:
            print(f"  ⚠️  Only in dataset 2: {only_in_2}")
    
    return common


def train_drilling_models(train_data, test_data=None, rop_params=None, torque_params=None):
    """
    Train both ROP and Torque models using PyTorch
    Works with any combination of available columns
    
    Args:
        train_data: Training dataset
        test_data: Optional separate test dataset
        rop_params: Optimized hyperparameters for ROP model
        torque_params: Optimized hyperparameters for Torque model
    """
    from drilling_ann import DrillingPredictor
    
    # Determine available feature columns dynamically
    possible_features = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    feature_columns = [col for col in possible_features if col in train_data.columns]
    
    print(f"\n🎯 Using {len(feature_columns)} features: {feature_columns}")
    
    # Check if target columns exist
    has_rop = 'ROP' in train_data.columns
    has_torque = 'Surface_Torque' in train_data.columns
    
    if not has_rop and not has_torque:
        raise ValueError("❌ Neither ROP nor Surface_Torque found in training data!")
    
    results = {}
    
    # ----- ROP model -----
    if has_rop:
        print("\n" + "="*80)
        print("🔧 Training ROP Model...")
        print("="*80)
        
        X_train = train_data[feature_columns].values
        y_rop_train = train_data['ROP'].values
        
        # Handle test data
        if test_data is not None and 'ROP' in test_data.columns:
            print("  Using separate TEST dataset")
            # Check if test data has all required features
            test_features = [col for col in feature_columns if col in test_data.columns]
            if len(test_features) != len(feature_columns):
                print(f"  ⚠️  TEST data missing features: {set(feature_columns) - set(test_features)}")
                print(f"  ⚠️  Falling back to train/val split")
                test_data_rop = None
            else:
                X_test = test_data[feature_columns].values
                y_rop_test = test_data['ROP'].values
                test_data_rop = test_data
        else:
            test_data_rop = None
        
        if test_data_rop is None:
            print("  Splitting TRAIN data (80-20)")
            X_train_full = X_train.copy()
            y_rop_train_full = y_rop_train.copy()
            X_train, X_test, y_rop_train, y_rop_test = train_test_split(
                X_train_full, y_rop_train_full, test_size=0.2, random_state=42
            )
        
        # Set hyperparameters for ROP
        if rop_params is not None:
            rop_params = list(rop_params)
            hidden_sizes = [int(rop_params[0])]
            dropout_rate = float(np.clip(rop_params[1], 0.2, 0.5))
            
            valid_lrs = [0.001, 0.0005, 0.0001]
            lr_index = int(np.clip(rop_params[2], 0, 2))
            learning_rate = valid_lrs[lr_index]
            
            batch_size = int(rop_params[3])
            weight_decay = float(rop_params[4])
        else:
            hidden_sizes = [8]
            learning_rate = 0.001
            batch_size = 4
            dropout_rate = 0.3
            weight_decay = 1e-4
        
        print(f"  Hyperparameters:")
        print(f"    Hidden neurons: {hidden_sizes[0]}")
        print(f"    Dropout:        {dropout_rate:.3f}")
        print(f"    Learning rate:  {learning_rate}")
        print(f"    Batch size:     {batch_size}")
        print(f"    Weight decay:   {weight_decay:.6f}")
        
        rop_model = DrillingPredictor(
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=200,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )
        
        X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
            X_train, y_rop_train, test_size=0.2, random_state=42
        )
        
        rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
        rop_results = rop_model.evaluate(X_test, y_rop_test)
        
        print(f"\n{'='*80}")
        print(f"✅ ROP Model Results:")
        print(f"  RMSE: {rop_results['rmse']:.4f}")
        print(f"  R²:   {rop_results['r2']:.4f}")
        print(f"  AARE: {rop_results['aare']:.2f}%")
        print(f"{'='*80}")
        
        results['rop_model'] = rop_model
        results['rop_results'] = rop_results
    else:
        print("\n⚠️  Skipping ROP model (ROP column not found)")
        results['rop_model'] = None
        results['rop_results'] = None
    
    # ----- Torque model -----
    if has_torque:
        print("\n" + "="*80)
        print("🔧 Training Torque Model...")
        print("="*80)
        
        X_train = train_data[feature_columns].values
        y_torque_train = train_data['Surface_Torque'].values
        
        # Handle test data
        if test_data is not None and 'Surface_Torque' in test_data.columns:
            print("  Using separate TEST dataset")
            test_features = [col for col in feature_columns if col in test_data.columns]
            if len(test_features) != len(feature_columns):
                print(f"  ⚠️  TEST data missing features: {set(feature_columns) - set(test_features)}")
                print(f"  ⚠️  Falling back to train/val split")
                test_data_torque = None
            else:
                X_test = test_data[feature_columns].values
                y_torque_test = test_data['Surface_Torque'].values
                test_data_torque = test_data
        else:
            test_data_torque = None
        
        if test_data_torque is None:
            print("  Splitting TRAIN data (80-20)")
            X_train_full = X_train.copy()
            y_torque_train_full = y_torque_train.copy()
            X_train, X_test, y_torque_train, y_torque_test = train_test_split(
                X_train_full, y_torque_train_full, test_size=0.2, random_state=42
            )
        
        # Set hyperparameters for Torque
        if torque_params is not None:
            torque_params = list(torque_params)
            hidden_sizes = [int(torque_params[0])]
            dropout_rate = float(np.clip(torque_params[1], 0.2, 0.5))
            
            valid_lrs = [0.001, 0.0005, 0.0001]
            lr_index = int(np.clip(torque_params[2], 0, 2))
            learning_rate = valid_lrs[lr_index]
            
            batch_size = int(torque_params[3])
            weight_decay = float(torque_params[4])
        else:
            hidden_sizes = [8]
            learning_rate = 0.001
            batch_size = 4
            dropout_rate = 0.3
            weight_decay = 1e-4
        
        print(f"  Hyperparameters:")
        print(f"    Hidden neurons: {hidden_sizes[0]}")
        print(f"    Dropout:        {dropout_rate:.3f}")
        print(f"    Learning rate:  {learning_rate}")
        print(f"    Batch size:     {batch_size}")
        print(f"    Weight decay:   {weight_decay:.6f}")
        
        torque_model = DrillingPredictor(
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=200,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )
        
        X_train_split, X_val_split, y_torque_train_split, y_torque_val_split = train_test_split(
            X_train, y_torque_train, test_size=0.2, random_state=42
        )
        
        torque_model.train(X_train_split, y_torque_train_split, X_val_split, y_torque_val_split)
        torque_results = torque_model.evaluate(X_test, y_torque_test)
        
        print(f"\n{'='*80}")
        print(f"✅ Torque Model Results:")
        print(f"  RMSE: {torque_results['rmse']:.4f}")
        print(f"  R²:   {torque_results['r2']:.4f}")
        print(f"  AARE: {torque_results['aare']:.2f}%")
        print(f"{'='*80}")
        
        results['torque_model'] = torque_model
        results['torque_results'] = torque_results
    else:
        print("\n⚠️  Skipping Torque model (Surface_Torque column not found)")
        results['torque_model'] = None
        results['torque_results'] = None
    
    # Collect predictions for plotting
    if results['rop_model'] is not None:
        # Get training predictions
        X_train_full = train_data[feature_columns].values
        y_rop_train_actual = train_data['ROP'].values
        y_rop_train_pred = results['rop_model'].predict(X_train_full)
        
        # Test predictions already available
        y_rop_test_actual = y_rop_test
        y_rop_test_pred = results['rop_results']['predictions']
        
        # Store for later use
        results['rop_train_actual'] = y_rop_train_actual
        results['rop_train_pred'] = y_rop_train_pred
        results['rop_test_actual'] = y_rop_test_actual
        results['rop_test_pred'] = y_rop_test_pred
    
    if results['torque_model'] is not None:
        # Get training predictions
        X_train_full = train_data[feature_columns].values
        y_torque_train_actual = train_data['Surface_Torque'].values
        y_torque_train_pred = results['torque_model'].predict(X_train_full)
        
        # Test predictions already available
        y_torque_test_actual = y_torque_test
        y_torque_test_pred = results['torque_results']['predictions']
        
        # Store for later use
        results['torque_train_actual'] = y_torque_train_actual
        results['torque_train_pred'] = y_torque_train_pred
        results['torque_test_actual'] = y_torque_test_actual
        results['torque_test_pred'] = y_torque_test_pred
    
    # Generate plots
    print("\n" + "="*80)
    print("📊 Generating Prediction Comparison Plots...")
    print("="*80)
    
    if results['rop_model'] is not None and results['torque_model'] is not None:
        # Plot all 4 subplots
        plot_both_models_comparison(
            results['rop_train_actual'], results['rop_train_pred'],
            results['rop_test_actual'], results['rop_test_pred'],
            results['torque_train_actual'], results['torque_train_pred'],
            results['torque_test_actual'], results['torque_test_pred'],
            save_path='results_comparison_ann.png'
        )
    else:
        # Plot individual models
        if results['rop_model'] is not None:
            plot_prediction_comparison(
                results['rop_train_actual'], results['rop_train_pred'],
                results['rop_test_actual'], results['rop_test_pred'],
                target_name='ROP',
                save_path='results_comparison_rop_ann.png'
            )
        
        if results['torque_model'] is not None:
            plot_prediction_comparison(
                results['torque_train_actual'], results['torque_train_pred'],
                results['torque_test_actual'], results['torque_test_pred'],
                target_name='Torque',
                save_path='results_comparison_torque_ann.png'
            )
    
    return results


def main():
    """
    Main function - FLEXIBLE version
    Works with ANY Excel file as train or test
    """
    print("\n" + "="*80)
    print("🚀 DRILLING PARAMETER OPTIMIZATION SYSTEM")
    print("   SMART AUTO-HEADER DETECTION")
    print("="*80)
    
    # ===== تنظیمات - فقط اینجا رو عوض کنید =====
    train_file = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#RR#34.xlsx"  # فایل train
    test_file = r"E:\Data\pure\drill operation\human edit\Bit Data#1214#RR#34.xlsx"                          # فایل test (یا None)
    
    columns_to_drop = ['Lit', 'Date', 'Time']  # ستون‌هایی که می‌خواید حذف بشه
    # مثال: columns_to_drop = ['Time', 'Date', 'HOB']
    
    # ===== بارگذاری داده TRAIN =====
    print("\n" + "="*80)
    print("📚 LOADING TRAIN DATA")
    print("="*80)
    
    try:
        train_data = load_drilling_data(train_file, columns_to_drop=columns_to_drop, verbose=True)
    except Exception as e:
        print(f"\n❌ Failed to load TRAIN data: {e}")
        return None
    
    # ===== بارگذاری داده TEST (اختیاری) =====
    test_data = None
    if test_file is not None:
        print("\n" + "="*80)
        print("📚 LOADING TEST DATA")
        print("="*80)
        
        try:
            test_data = load_drilling_data(test_file, columns_to_drop=columns_to_drop, verbose=True)
            
            # پیدا کردن ستون‌های مشترک
            common_cols = find_common_columns(train_data, test_data, verbose=True)
            
            if len(common_cols) == 0:
                print("\n⚠️  WARNING: No common columns! Using only TRAIN data.")
                test_data = None
            else:
                # نگه داشتن فقط ستون‌های مشترک
                print(f"\n✂️  Keeping only {len(common_cols)} common columns")
                train_data = train_data[common_cols]
                test_data = test_data[common_cols]
                
                print(f"  TRAIN shape: {train_data.shape}")
                print(f"  TEST shape:  {test_data.shape}")
        
        except Exception as e:
            print(f"\n⚠️  Failed to load TEST data: {e}")
            print(f"  Continuing with TRAIN data only...")
            test_data = None
    
    # ===== آموزش مدل‌های پایه =====
    print("\n" + "="*80)
    print("📊 [STEP 1] Training Baseline Models...")
    print("="*80)
    
    try:
        from baseline_models import train_baseline_models
        baseline_results = train_baseline_models(train_data)
    except Exception as e:
        print(f"⚠️  Baseline models failed: {e}")
        baseline_results = None
    
    # ===== بهینه‌سازی هایپرپارامترها =====
    print("\n" + "="*80)
    print("⚙️  [STEP 2] Optimizing Hyperparameters...")
    print("="*80)
    
    best_rop_params = None
    best_torque_params = None
    
    try:
        from ann_hyperparameter_optimizer import optimize_ann_hyperparameters
        
        if 'ROP' in train_data.columns:
            print("\n🎯 Optimizing ROP model...")
            best_rop_params = optimize_ann_hyperparameters(
                train_data,
                target_column='ROP',
                pop_size=10,
                max_iter=20
            )
        else:
            print("\n⚠️  Skipping ROP optimization (column not found)")
        
        if 'Surface_Torque' in train_data.columns:
            print("\n🎯 Optimizing Torque model...")
            best_torque_params = optimize_ann_hyperparameters(
                train_data,
                target_column='Surface_Torque',
                pop_size=10,
                max_iter=20
            )
        else:
            print("\n⚠️  Skipping Torque optimization (column not found)")
    
    except Exception as e:
        print(f"⚠️  Hyperparameter optimization failed: {e}")
    
    # ===== آموزش مدل‌های نهایی =====
    print("\n" + "="*80)
    print("🏋️  [STEP 3] Training Final Models...")
    print("="*80)
    
    try:
        model_results = train_drilling_models(
            train_data,
            test_data=test_data,
            rop_params=best_rop_params,
            torque_params=best_torque_params
        )
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ===== نتایج نهایی =====
    print("\n" + "="*80)
    print("🎉 FINAL RESULTS")
    print("="*80)
    
    print(f"\n📁 Data Files:")
    print(f"  TRAIN: {os.path.basename(train_file)} ({len(train_data)} samples)")
    if test_data is not None:
        print(f"  TEST:  {os.path.basename(test_file)} ({len(test_data)} samples)")
    else:
        print(f"  TEST:  Using train/val split")
    
    print(f"\n📊 Model Performance:")
    
    if model_results.get('rop_results'):
        rop_res = model_results['rop_results']
        print(f"  ✓ ROP Model:")
        print(f"      R²:   {rop_res['r2']:.4f}")
        print(f"      RMSE: {rop_res['rmse']:.4f}")
        print(f"      AARE: {rop_res['aare']:.2f}%")
    
    if model_results.get('torque_results'):
        torque_res = model_results['torque_results']
        print(f"  ✓ Torque Model:")
        print(f"      R²:   {torque_res['r2']:.4f}")
        print(f"      RMSE: {torque_res['rmse']:.4f}")
        print(f"      AARE: {torque_res['aare']:.2f}%")
    
    print("\n" + "="*80)
    print("✅ Optimization completed successfully!")
    print("="*80)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'baseline_results': baseline_results,
        **model_results
    }


if __name__ == "__main__":
    results = main()
