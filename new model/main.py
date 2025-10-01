# main.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import differential_evolution

# XGBoost imports (replacing ANN imports)
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

    # ⭐ محاسبه میانگین ⭐ Handle range patterns like "20-25" or "20 - 25"
    range_pattern = r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
    match = re.search(range_pattern, value_str)
    if match:
        try:
            num1 = float(match.group(1))
            num2 = float(match.group(2))
            return (num1 + num2) / 2.0
        except (ValueError, TypeError):
            pass

    # Extract first number found
    number_pattern = r'(\d+(?:\.\d+)?)'
    match = re.search(number_pattern, value_str)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass

    # If all else fails
    print(f"    ⚠️  Could not convert '{value}' to number. Using NaN.")
    return np.nan

def standardize_column_names(df):
    """
    تبدیل نام‌های مختلف ستون‌ها به فرمت استاندارد
    ⭐ FIX: Also handles duplicate columns by keeping only the first occurrence ⭐
    """
    # Comprehensive column name mapping
    column_mappings = {
        # Depth variations
        'Depth': ['depth', 'md', 'measured depth', 'measured_depth', 'depth_m', 'depth_ft'],
        
        # WOB variations
        'WOB': ['wob', 'weight on bit', 'mob', 'weight_on_bit', 'bit_weight', 'wob_klb', 'wob_klbs'],
        
        # ROP variations
        'ROP': ['rop', 'rate of penetration', 'penetration rate', 'drilling rate', 'rate_of_penetration'],
        
        # RPM variations
        'RPM': ['rpm', 'rotary speed', 'rotation', 'rotary_speed', 'rotation_speed'],
        
        # Torque variations
        'Surface_Torque': ['torque', 'surface torque', 'surface_torque', 'rot_torque', 'rotary_torque'],
        
        # Flow rate (Q) variations
        'Q': ['q', 'flow', 'flow rate', 'flow_rate', 'gpm', 'pump_rate', 'total_flow'],
        
        # SPP variations
        'SPP': ['spp', 'standpipe pressure', 'standpipe_pressure', 'pump_pressure', 'surface_pressure'],
        
        # Hook Load variations
        'Hook_Load': ['hook load', 'hook_load', 'hookload', 'hook', 'weight_indicator'],
        
        # Mud Weight variations
        'MW': ['mw', 'mud weight', 'mud_weight', 'density', 'mud_density'],
        
        # Viscosity variations
        'Viscosity': ['vis', 'viscosity', 'pv', 'plastic_viscosity', 'funnel_viscosity']
    }

    # Create reverse mapping
    reverse_map = {}
    for standard_name, variations in column_mappings.items():
        for var in variations:
            reverse_map[var.lower().strip()] = standard_name

    # ⭐ NEW: Track which standard names we've already mapped ⭐
    already_mapped = set()

    # Rename columns
    new_columns = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check if this column matches any variation
        if col_lower in reverse_map:
            standard_name = reverse_map[col_lower]
            
            # ⭐ FIX: Only map if we haven't seen this standard name before ⭐
            if standard_name not in already_mapped:
                new_columns[col] = standard_name
                already_mapped.add(standard_name)
            else:
                # If duplicate, keep original name with suffix
                new_columns[col] = f"{col}_duplicate"
        else:
            new_columns[col] = col

    df_renamed = df.rename(columns=new_columns)
    
    # Print mapping results
    print("\n📋 Column name standardization:")
    for old, new in new_columns.items():
        if old != new:
            print(f"  '{old}' → '{new}'")
    
    return df_renamed

def load_drilling_data(file_path, verbose=True):
    """
    🚀 Universal data loader که به صورت اتوماتیک header رو پیدا می‌کنه
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📂 Loading: {os.path.basename(file_path)}")
        print(f"{'='*70}")

    # Detect header row automatically
    header_row = detect_header_row(file_path)

    # Load with detected header
    df = pd.read_excel(file_path, header=header_row)

    if verbose:
        print(f"\n✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        print(f"📊 Original columns: {list(df.columns)}")

    # Standardize column names
    df = standardize_column_names(df)

    if verbose:
        print(f"\n🔧 Standardized columns: {list(df.columns)}")

    # Clean numeric columns
    if verbose:
        print(f"\n🧹 Cleaning numeric data...")

    for col in df.columns:
        if col not in ['Date', 'Time', 'Formation']:  # Skip non-numeric columns
            df[col] = clean_numeric_column(df[col])

    # Drop rows where ALL values are NaN
    df_cleaned = df.dropna(how='all')

    if verbose:
        rows_dropped = len(df) - len(df_cleaned)
        print(f"\n🗑️  Dropped {rows_dropped} completely empty rows")
        print(f"✅ Final dataset: {len(df_cleaned)} rows × {len(df_cleaned.columns)} columns")

    return df_cleaned

def visualize_data(data, title_prefix=""):
    """Create comprehensive visualizations of drilling data"""
    
    # Check required columns
    required_cols = ['Depth', 'WOB', 'ROP', 'RPM', 'Surface_Torque', 'SPP', 'Q']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"⚠️  Missing columns for visualization: {missing_cols}")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}Drilling Parameters Analysis', fontsize=16, fontweight='bold')
    
    # 1. ROP vs Depth
    axes[0, 0].plot(data['ROP'], data['Depth'], 'b-', linewidth=1)
    axes[0, 0].set_xlabel('ROP (m/hr)', fontsize=10)
    axes[0, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 0].set_title('Rate of Penetration vs Depth', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. WOB vs Depth
    axes[0, 1].plot(data['WOB'], data['Depth'], 'r-', linewidth=1)
    axes[0, 1].set_xlabel('WOB (klbs)', fontsize=10)
    axes[0, 1].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 1].set_title('Weight on Bit vs Depth', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RPM vs Depth
    axes[1, 0].plot(data['RPM'], data['Depth'], 'g-', linewidth=1)
    axes[1, 0].set_xlabel('RPM', fontsize=10)
    axes[1, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[1, 0].set_title('Rotary Speed vs Depth', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Torque vs Depth
    axes[1, 1].plot(data['Surface_Torque'], data['Depth'], 'm-', linewidth=1)
    axes[1, 1].set_xlabel('Surface Torque (N.m)', fontsize=10)
    axes[1, 1].set_ylabel('Depth (m)', fontsize=10)
    axes[1, 1].set_title('Surface Torque vs Depth', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. ROP vs WOB scatter
    axes[2, 0].scatter(data['WOB'], data['ROP'], alpha=0.5, s=10)
    axes[2, 0].set_xlabel('WOB (klbs)', fontsize=10)
    axes[2, 0].set_ylabel('ROP (m/hr)', fontsize=10)
    axes[2, 0].set_title('ROP vs WOB Relationship', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. ROP vs RPM scatter
    axes[2, 1].scatter(data['RPM'], data['ROP'], alpha=0.5, s=10, c='orange')
    axes[2, 1].set_xlabel('RPM', fontsize=10)
    axes[2, 1].set_ylabel('ROP (m/hr)', fontsize=10)
    axes[2, 1].set_title('ROP vs RPM Relationship', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_dict):
    """
    Compare different models' performance
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results dicts as values
        Example: {'Ridge': {...}, 'Random Forest': {...}, 'XGBoost': {...}}
    """
    models = list(results_dict.keys())
    r2_scores = [results_dict[model]['r2'] for model in models]
    rmse_scores = [results_dict[model]['rmse'] for model in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² comparison
    axes[0].bar(models, r2_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # RMSE comparison
    axes[1].bar(models, rmse_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(rmse_scores):
        axes[1].text(i, v + max(rmse_scores)*0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, model_name, target_name='ROP'):
    """
    Plot predicted vs actual values
    
    Parameters:
    -----------
    y_true : array, actual values
    y_pred : array, predicted values
    model_name : str, name of the model
    target_name : str, name of target variable
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel(f'Actual {target_name}', fontsize=12)
    axes[0].set_ylabel(f'Predicted {target_name}', fontsize=12)
    axes[0].set_title(f'{model_name} - Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel(f'Predicted {target_name}', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def optimize_drilling_parameters(data, rop_model, torque_model, 
                                 fixed_spp=None, fixed_q=None, 
                                 fixed_depth=None, fixed_hook_load=None):
    """
    Optimize WOB and RPM for maximum ROP while keeping torque in acceptable range
    
    Parameters:
    -----------
    data : DataFrame
        Drilling data for extracting typical parameter ranges
    rop_model : trained XGBoost model for ROP prediction
    torque_model : trained XGBoost model for torque prediction
    fixed_spp : float, fixed standpipe pressure (if None, uses median)
    fixed_q : float, fixed flow rate (if None, uses median)
    fixed_depth : float, fixed depth (if None, uses median)
    fixed_hook_load : float, fixed hook load (if None, uses median)
    
    Returns:
    --------
    results : dict with optimization results
    """
    
    # Set fixed parameters to median values if not provided
    if fixed_spp is None:
        fixed_spp = data['SPP'].median()
    if fixed_q is None:
        fixed_q = data['Q'].median()
    if fixed_depth is None:
        fixed_depth = data['Depth'].median()
    if fixed_hook_load is None:
        fixed_hook_load = data['Hook_Load'].median()
    
    fixed_params = np.array([fixed_spp, fixed_q, fixed_depth, fixed_hook_load])
    
    print(f"\n{'='*70}")
    print("Fixed Parameters for Optimization:")
    print(f"{'='*70}")
    print(f"  SPP:        {fixed_spp:.2f}")
    print(f"  Q:          {fixed_q:.2f}")
    print(f"  Depth:      {fixed_depth:.2f}")
    print(f"  Hook Load:  {fixed_hook_load:.2f}")
    print(f"{'='*70}")
    
    # Define bounds for WOB and RPM based on data
    wob_min, wob_max = data['WOB'].quantile([0.05, 0.95])
    rpm_min, rpm_max = data['RPM'].quantile([0.05, 0.95])
    
    bounds = [
        (wob_min, wob_max),  # WOB bounds
        (rpm_min, rpm_max)   # RPM bounds
    ]
    
    print(f"\nOptimization Bounds:")
    print(f"  WOB: [{wob_min:.2f}, {wob_max:.2f}]")
    print(f"  RPM: [{rpm_min:.2f}, {rpm_max:.2f}]")
    
    # Run Differential Evolution optimization
    de = DifferentialEvolution(pop_size=30, F=0.5, CR=0.7, max_iter=100)
    results = de.optimize(bounds, rop_model, torque_model, fixed_params, verbose=True)
    
    # Plot convergence
    de.plot_convergence(results['fitness_history'])
    
    return results

def compare_optimization_methods(data, rop_model, torque_model, fixed_params):
    """
    Compare different optimization methods (DE, PSO, GA, etc.)
    This is a placeholder for future implementation
    """
    # TODO: Implement comparison of different metaheuristic algorithms
    pass

def sensitivity_analysis(data, rop_model, torque_model, optimal_wob, optimal_rpm):
    """
    Perform sensitivity analysis on optimized parameters
    
    Parameters:
    -----------
    data : DataFrame
    rop_model : trained model
    torque_model : trained model
    optimal_wob : float, optimized WOB
    optimal_rpm : float, optimized RPM
    """
    fixed_spp = data['SPP'].median()
    fixed_q = data['Q'].median()
    fixed_depth = data['Depth'].median()
    fixed_hook_load = data['Hook_Load'].median()
    
    # Vary WOB while keeping RPM constant
    wob_range = np.linspace(optimal_wob * 0.7, optimal_wob * 1.3, 50)
    rop_wob = []
    torque_wob = []
    
    for wob in wob_range:
        params = np.array([[wob, optimal_rpm, fixed_spp, fixed_q, fixed_depth, fixed_hook_load]])
        rop_wob.append(rop_model.predict(params)[0])
        torque_wob.append(torque_model.predict(params)[0] * 0.737562)  # Convert to Lb.Ft
    
    # Vary RPM while keeping WOB constant
    rpm_range = np.linspace(optimal_rpm * 0.7, optimal_rpm * 1.3, 50)
    rop_rpm = []
    torque_rpm = []
    
    for rpm in rpm_range:
        params = np.array([[optimal_wob, rpm, fixed_spp, fixed_q, fixed_depth, fixed_hook_load]])
        rop_rpm.append(rop_model.predict(params)[0])
        torque_rpm.append(torque_model.predict(params)[0] * 0.737562)
    
    # Plot sensitivity analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # ROP vs WOB
    axes[0, 0].plot(wob_range, rop_wob, 'b-', linewidth=2)
    axes[0, 0].axvline(optimal_wob, color='r', linestyle='--', label='Optimal WOB')
    axes[0, 0].set_xlabel('WOB (klbs)', fontsize=12)
    axes[0, 0].set_ylabel('ROP (m/hr)', fontsize=12)
    axes[0, 0].set_title('ROP Sensitivity to WOB', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Torque vs WOB
    axes[0, 1].plot(wob_range, torque_wob, 'r-', linewidth=2)
    axes[0, 1].axvline(optimal_wob, color='r', linestyle='--', label='Optimal WOB')
    axes[0, 1].axhline(13000, color='g', linestyle='--', label='Min Torque')
    axes[0, 1].axhline(19000, color='g', linestyle='--', label='Max Torque')
    axes[0, 1].set_xlabel('WOB (klbs)', fontsize=12)
    axes[0, 1].set_ylabel('Torque (Lb.Ft)', fontsize=12)
    axes[0, 1].set_title('Torque Sensitivity to WOB', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROP vs RPM
    axes[1, 0].plot(rpm_range, rop_rpm, 'b-', linewidth=2)
    axes[1, 0].axvline(optimal_rpm, color='r', linestyle='--', label='Optimal RPM')
    axes[1, 0].set_xlabel('RPM', fontsize=12)
    axes[1, 0].set_ylabel('ROP (m/hr)', fontsize=12)
    axes[1, 0].set_title('ROP Sensitivity to RPM', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Torque vs RPM
    axes[1, 1].plot(rpm_range, torque_rpm, 'r-', linewidth=2)
    axes[1, 1].axvline(optimal_rpm, color='r', linestyle='--', label='Optimal RPM')
    axes[1, 1].axhline(13000, color='g', linestyle='--', label='Min Torque')
    axes[1, 1].axhline(19000, color='g', linestyle='--', label='Max Torque')
    axes[1, 1].set_xlabel('RPM', fontsize=12)
    axes[1, 1].set_ylabel('Torque (Lb.Ft)', fontsize=12)
    axes[1, 1].set_title('Torque Sensitivity to RPM', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_report(data, baseline_results, xgboost_results, optimization_results):
    """
    Generate a comprehensive text report of all results
    
    Parameters:
    -----------
    data : DataFrame
    baseline_results : dict, results from baseline models
    xgboost_results : dict, results from XGBoost models
    optimization_results : dict, results from DE optimization
    """
    report = []
    report.append("=" * 80)
    report.append("DRILLING PARAMETER OPTIMIZATION - COMPREHENSIVE REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Dataset Summary
    report.append("1. DATASET SUMMARY")
    report.append("-" * 80)
    report.append(f"   Total Records: {len(data)}")
    report.append(f"   Depth Range: {data['Depth'].min():.2f} - {data['Depth'].max():.2f} m")
    report.append(f"   WOB Range: {data['WOB'].min():.2f} - {data['WOB'].max():.2f} klbs")
    report.append(f"   RPM Range: {data['RPM'].min():.2f} - {data['RPM'].max():.2f}")
    report.append(f"   ROP Range: {data['ROP'].min():.2f} - {data['ROP'].max():.2f} m/hr")
    report.append(f"   Torque Range: {data['Surface_Torque'].min():.2f} - {data['Surface_Torque'].max():.2f} N.m")
    report.append("")
    
    # Model Performance Comparison
    report.append("2. MODEL PERFORMANCE COMPARISON")
    report.append("-" * 80)
    report.append("   ROP PREDICTION:")
    report.append(f"      Ridge Regression    - R²: {baseline_results['rop_ridge_r2']:.4f}, RMSE: {baseline_results['rop_ridge_rmse']:.4f}")
    report.append(f"      Random Forest       - R²: {baseline_results['rop_rf_r2']:.4f}, RMSE: {baseline_results['rop_rf_rmse']:.4f}")
    report.append(f"      XGBoost (Optimized) - R²: {xgboost_results[0]['r2']:.4f}, RMSE: {xgboost_results[0]['rmse']:.4f}")
    report.append("")
    report.append("   TORQUE PREDICTION:")
    report.append(f"      Ridge Regression    - R²: {baseline_results['torque_ridge_r2']:.4f}, RMSE: {baseline_results['torque_ridge_rmse']:.4f}")
    report.append(f"      Random Forest       - R²: {baseline_results['torque_rf_r2']:.4f}, RMSE: {baseline_results['torque_rf_rmse']:.4f}")
    report.append(f"      XGBoost (Optimized) - R²: {xgboost_results[1]['r2']:.4f}, RMSE: {xgboost_results[1]['rmse']:.4f}")
    report.append("")
    
    # Optimization Results
    report.append("3. OPTIMIZATION RESULTS")
    report.append("-" * 80)
    report.append(f"   Optimal WOB:        {optimization_results['optimal_wob']:.2f} klbs")
    report.append(f"   Optimal RPM:        {optimization_results['optimal_rpm']:.2f}")
    report.append(f"   Predicted ROP:      {optimization_results['predicted_rop']:.2f} m/hr")
    report.append(f"   Predicted Torque:   {optimization_results['predicted_torque']:.2f} Lb.Ft")
    report.append("")
    
    # Recommendations
    report.append("4. RECOMMENDATIONS")
    report.append("-" * 80)
    
    # Check if torque is within acceptable range
    torque = optimization_results['predicted_torque']
    if 13000 <= torque <= 19000:
        report.append("   ✅ Optimized parameters result in torque within acceptable range")
    elif torque < 13000:
        report.append("   ⚠️  Torque below minimum threshold - consider increasing WOB or RPM")
    else:
        report.append("   ⚠️  Torque above maximum threshold - consider reducing WOB or RPM")
    
    # ROP improvement
    current_avg_rop = data['ROP'].mean()
    rop_improvement = ((optimization_results['predicted_rop'] - current_avg_rop) / current_avg_rop) * 100
    report.append(f"   Expected ROP improvement: {rop_improvement:.1f}% over current average")
    report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Print and save report
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    with open('drilling_optimization_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n📄 Report saved to: drilling_optimization_report.txt")

def main():
    """
    Main execution function for drilling parameter optimization using XGBoost
    """
    print("\n" + "="*80)
    print("DRILLING PARAMETER OPTIMIZATION WITH XGBOOST")
    print("="*80)
    
    # Step 1: Load data
    file_path = input("\nEnter the path to your Excel file: ").strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return
    
    try:
        data = load_drilling_data(file_path)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Check for required columns
    required_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load', 'ROP', 'Surface_Torque']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"\n❌ Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(data.columns)}")
        return
    
    # Remove rows with missing values in required columns
    data_clean = data[required_columns].dropna()
    print(f"\n✅ Clean dataset: {len(data_clean)} rows (removed {len(data) - len(data_clean)} rows with missing values)")
    
    # Step 2: Visualize data
    print("\n" + "="*80)
    print("STEP 1: DATA VISUALIZATION")
    print("="*80)
    visualize_data(data_clean)
    
    # Step 3: Train baseline models
    print("\n" + "="*80)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("="*80)
    baseline_results = train_baseline_models(data_clean)
    
    # Step 4: Hyperparameter optimization for XGBoost
    print("\n" + "="*80)
    print("STEP 3: XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    optimize_hyperparams = input("\nDo you want to optimize XGBoost hyperparameters? (yes/no): ").strip().lower()
    
    if optimize_hyperparams in ['yes', 'y']:
        print("\n🔍 Optimizing ROP model hyperparameters...")
        best_rop_params = optimize_xgboost_hyperparameters(
            data_clean, 
            target_column='ROP',
            pop_size=20,
            max_iter=30
        )
        
        print("\n🔍 Optimizing Torque model hyperparameters...")
        best_torque_params = optimize_xgboost_hyperparameters(
            data_clean,
            target_column='Surface_Torque',
            pop_size=20,
            max_iter=30
        )
        
        # Train final models with optimized hyperparameters
        print("\n" + "="*80)
        print("STEP 4: TRAINING OPTIMIZED XGBOOST MODELS")
        print("="*80)
        
        # Prepare data
        feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
        X = data_clean[feature_columns].values
        
        # Train ROP model with optimized params
        from sklearn.model_selection import train_test_split
        y_rop = data_clean['ROP'].values
        X_train, X_test, y_rop_train, y_rop_test = train_test_split(
            X, y_rop, test_size=0.2, random_state=42
        )
        
        rop_model = DrillingXGBoostPredictor(
            n_estimators=int(best_rop_params[0]),
            max_depth=int(best_rop_params[1]),
            learning_rate=best_rop_params[2],
            subsample=best_rop_params[3],
            colsample_bytree=best_rop_params[4],
            gamma=best_rop_params[5],
            min_child_weight=int(best_rop_params[6]),
            reg_alpha=best_rop_params[7],
            reg_lambda=best_rop_params[8]
        )
        
        X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
            X_train, y_rop_train, test_size=0.2, random_state=42
        )
        rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
        rop_results = rop_model.evaluate(X_test, y_rop_test)
        
        # Train Torque model with optimized params
        y_torque = data_clean['Surface_Torque'].values
        X_train, X_test, y_torque_train, y_torque_test = train_test_split(
            X, y_torque, test_size=0.2, random_state=42
        )
        
        torque_model = DrillingXGBoostPredictor(
            n_estimators=int(best_torque_params[0]),
            max_depth=int(best_torque_params[1]),
            learning_rate=best_torque_params[2],
            subsample=best_torque_params[3],
            colsample_bytree=best_torque_params[4],
            gamma=best_torque_params[5],
            min_child_weight=int(best_torque_params[6]),
            reg_alpha=best_torque_params[7],
            reg_lambda=best_torque_params[8]
        )
        
        X_train_split, X_val_split, y_torque_train_split, y_torque_val_split = train_test_split(
            X_train, y_torque_train, test_size=0.2, random_state=42
        )
        torque_model.train(X_train_split, y_torque_train_split, X_val_split, y_torque_val_split)
        torque_results = torque_model.evaluate(X_test, y_torque_test)
        
    else:
        # Train with default parameters
        print("\n" + "="*80)
        print("STEP 4: TRAINING XGBOOST MODELS (Default Parameters)")
        print("="*80)
        rop_model, torque_model, rop_results, torque_results = train_drilling_models(data_clean)
    
    # Step 5: Compare models
    print("\n" + "="*80)
    print("STEP 5: MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    rop_comparison = {
        'Ridge': {'r2': baseline_results['rop_ridge_r2'], 'rmse': baseline_results['rop_ridge_rmse']},
        'Random Forest': {'r2': baseline_results['rop_rf_r2'], 'rmse': baseline_results['rop_rf_rmse']},
        'XGBoost': {'r2': rop_results['r2'], 'rmse': rop_results['rmse']}
    }
    
    torque_comparison = {
        'Ridge': {'r2': baseline_results['torque_ridge_r2'], 'rmse': baseline_results['torque_ridge_rmse']},
        'Random Forest': {'r2': baseline_results['torque_rf_r2'], 'rmse': baseline_results['torque_rf_rmse']},
        'XGBoost': {'r2': torque_results['r2'], 'rmse': torque_results['rmse']}
    }
    
    print("\n📊 ROP Model Comparison:")
    plot_model_comparison(rop_comparison)
    
    print("\n📊 Torque Model Comparison:")
    plot_model_comparison(torque_comparison)
    
    # Step 6: Plot predictions
    print("\n" + "="*80)
    print("STEP 6: PREDICTION VISUALIZATION")
    print("="*80)
    
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data_clean[feature_columns].values
    y_rop = data_clean['ROP'].values
    y_torque = data_clean['Surface_Torque'].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(X, y_rop, test_size=0.2, random_state=42)
    X_train, X_test, y_torque_train, y_torque_test = train_test_split(X, y_torque, test_size=0.2, random_state=42)
    
    plot_predictions_vs_actual(y_rop_test, rop_results['predictions'], 'XGBoost', 'ROP')
    plot_predictions_vs_actual(y_torque_test, torque_results['predictions'], 'XGBoost', 'Torque')
    
    # Step 7: Optimize drilling parameters
    print("\n" + "="*80)
    print("STEP 7: DRILLING PARAMETER OPTIMIZATION")
    print("="*80)
    
    optimization_results = optimize_drilling_parameters(
        data_clean, rop_model, torque_model
    )
    
    # Step 8: Sensitivity analysis
    print("\n" + "="*80)
    print("STEP 8: SENSITIVITY ANALYSIS")
    print("="*80)
    
    sensitivity_analysis(
        data_clean, 
        rop_model, 
        torque_model,
        optimization_results['optimal_wob'],
        optimization_results['optimal_rpm']
    )
    
    # Step 9: Generate comprehensive report
    print("\n" + "="*80)
    print("STEP 9: GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    generate_report(
        data_clean,
        baseline_results,
        (rop_results, torque_results),
        optimization_results
    )
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nThank you for using the Drilling Parameter Optimization System!")

if __name__ == "__main__":
    main()
