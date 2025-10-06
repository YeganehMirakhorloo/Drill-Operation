"""
Plotting utilities for drilling prediction models
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


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
