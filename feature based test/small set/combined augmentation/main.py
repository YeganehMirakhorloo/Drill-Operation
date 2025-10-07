import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
from drilling_ann import train_drilling_models, DrillingPredictor
from ann_hyperparameter_optimizer import optimize_ann_hyperparameters
from data_augmentation import augment_training_data

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(filepath):
    """Load and prepare drilling data"""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    data = pd.read_csv(filepath)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        print(f"\nMissing values found:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values found")
    
    return data

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate AARE
    mask = y_true != 0
    if np.any(mask):
        aare = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        aare = np.inf
    
    return {'rmse': rmse, 'r2': r2, 'aare': aare}

def plot_feature_importance(model, feature_names, title, filename):
    """Plot feature importance for gradient boosting models"""
    importance = model.feature_importances_
    indices = np.argsort(importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance (gain)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved: {filename}")
    plt.close()

def train_gradient_boosting_rop(X_train, X_test, y_train, y_test, feature_names):
    """Train Gradient Boosting model for ROP prediction"""
    print("\n" + "="*80)
    print("TRAINING GRADIENT BOOSTING MODEL FOR ROP")
    print("="*80)
    
    # Augment training data
    print("\nAugmenting training data for Gradient Boosting...")
    X_train_aug, y_train_aug = augment_training_data(
        X_train, y_train,
        noise_levels=[0.05, 0.10, 0.20],
        verbose=True
    )
    
    # Train model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    
    print("Training Gradient Boosting model...")
    gb_model.fit(X_train_aug, y_train_aug)
    
    # Predictions on ORIGINAL (non-augmented) data
    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print(f"\nGradient Boosting ROP Results:")
    print(f"Train Set - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}, AARE: {train_metrics['aare']:.2f}%")
    print(f"Test Set  - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, AARE: {test_metrics['aare']:.2f}%")
    
    # Plot feature importance
    plot_feature_importance(gb_model, feature_names, 
                          'ROP Model - Feature Importance',
                          'rop_feature_importance.png')
    
    return gb_model, train_metrics, test_metrics, y_train_pred, y_test_pred

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, 
                    train_r2, test_r2, target_name, filename):
    """Plot actual vs predicted values"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, color='blue', edgecolors='k', s=50)
    axes[0].plot([y_train.min(), y_train.max()], 
                 [y_train.min(), y_train.max()], 
                 'k--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel(f'Real {target_name}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{target_name} Model - Training Set\nR² = {train_r2:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='red', edgecolors='k', s=50)
    axes[1].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'k--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel(f'Real {target_name}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{target_name} Model - Testing Set\nR² = {test_r2:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction plot saved: {filename}")
    plt.close()

def save_results_table(ann_train, ann_test, gb_train, gb_test, filename='final_report_rop.csv'):
    """Save comparison results to CSV (ROP only)"""
    results_df = pd.DataFrame({
        'Metric': ['RMSE', 'AARE (%)', 'R²'],
        'ANN (Train)': [
            ann_train['rmse'],
            ann_train['aare'],
            ann_train['r2']
        ],
        'ANN (Test)': [
            ann_test['rmse'],
            ann_test['aare'],
            ann_test['r2']
        ],
        'GB (Train)': [
            gb_train['rmse'],
            gb_train['aare'],
            gb_train['r2']
        ],
        'GB (Test)': [
            gb_test['rmse'],
            gb_test['aare'],
            gb_test['r2']
        ]
    })
    
    results_df.to_csv(filename, index=False)
    print(f"\n✅ Results table saved: {filename}")
    print("\nFinal Results Comparison (ROP):")
    print(results_df.to_string(index=False))
    
    return results_df

def plot_comparison_results(results_df, filename='results_comparison_rop.png'):
    """Plot comparison of ANN vs GB results"""
    metrics = results_df['Metric'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ann_train = results_df.loc[idx, 'ANN (Train)']
        ann_test = results_df.loc[idx, 'ANN (Test)']
        gb_train = results_df.loc[idx, 'GB (Train)']
        gb_test = results_df.loc[idx, 'GB (Test)']
        
        x = np.arange(2)
        width = 0.35
        
        axes[idx].bar(x - width/2, [ann_train, ann_test], width, 
                     label='ANN', color='steelblue', edgecolor='black')
        axes[idx].bar(x + width/2, [gb_train, gb_test], width, 
                     label='Gradient Boosting', color='coral', edgecolor='black')
        
        axes[idx].set_xlabel('Dataset', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'ROP - {metric} Comparison', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['Train', 'Test'])
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved: {filename}")
    plt.close()

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*20 + "ROP PREDICTION SYSTEM")
    print("="*80 + "\n")
    
    # Load data
    data = load_and_prepare_data('drilling_data.csv')
    
    # Define features
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    y_rop = data['ROP'].values
    
    # Split data
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # ==================== GRADIENT BOOSTING MODEL ====================
    gb_model, gb_train_metrics, gb_test_metrics, gb_y_train_pred, gb_y_test_pred = \
        train_gradient_boosting_rop(X_train, X_test, y_rop_train, y_rop_test, feature_columns)
    
    # Plot GB predictions
    plot_predictions(y_rop_train, gb_y_train_pred, y_rop_test, gb_y_test_pred,
                    gb_train_metrics['r2'], gb_test_metrics['r2'],
                    'ROP (m/hr)', 'gb_rop_predictions.png')
    
    # ==================== ANN MODEL ====================
    rop_model, rop_train_results, rop_test_results, X_train_ann, X_test_ann, y_rop_train_ann, y_rop_test_ann = \
        train_drilling_models(data)
    
    # Plot ANN predictions
    plot_predictions(y_rop_train_ann, rop_train_results['predictions'], 
                    y_rop_test_ann, rop_test_results['predictions'],
                    rop_train_results['r2'], rop_test_results['r2'],
                    'ROP (m/hr)', 'ann_rop_predictions.png')
    
    # ==================== SAVE RESULTS ====================
    results_df = save_results_table(
        rop_train_results, rop_test_results,
        gb_train_metrics, gb_test_metrics,
        filename='final_report_rop.csv'
    )
    
    # Plot comparison
    plot_comparison_results(results_df, filename='results_comparison_rop.png')
    
    print("\n" + "="*80)
    print(" "*25 + "✅ ALL DONE!")
    print("="*80 + "\n")
    
    return {
        'ann_model': rop_model,
        'gb_model': gb_model,
        'results': results_df
    }

if __name__ == "__main__":
    results = main()
