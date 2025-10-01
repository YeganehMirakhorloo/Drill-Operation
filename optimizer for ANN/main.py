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
from ann_hyperparameter_optimizer import optimize_ann_hyperparameters
from drilling_ann import DrillingPredictor




def convert_range_to_number(value):
    """
    Convert range strings like '15-25' to their midpoint (20.0)
    Handle various formats including single numbers, ranges, and non-numeric values
    """
    if pd.isna(value):
        return np.nan
    
    # Convert to string to handle different data types
    value_str = str(value).strip()
    
    # If it's already a number, return it
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle range patterns like "15-25", "15 - 25", "15to25", etc.
    range_patterns = [
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',  # "15-25" or "15 - 25"
        r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)',  # "15 to 25"
        r'(\d+(?:\.\d+)?)\s*~\s*(\d+(?:\.\d+)?)',   # "15 ~ 25"
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, value_str, re.IGNORECASE)
        if match:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            return (min_val + max_val) / 2.0  # Return midpoint
    
    # Handle single number with text like "15 kg", "25.5 rpm", etc.
    number_match = re.search(r'(\d+(?:\.\d+)?)', value_str)
    if number_match:
        return float(number_match.group(1))
    
    # Handle special cases
    if value_str.lower() in ['', '-', 'n/a', 'na', 'null', 'none']:
        return np.nan
    
    # If nothing matches, return NaN
    print(f"Warning: Could not convert '{value_str}' to number. Setting to NaN.")
    return np.nan

def clean_numeric_column(series):
    """
    Clean a pandas series containing mixed numeric and string data
    """
    return series.apply(convert_range_to_number)

def load_drilling_data(file_path=None):
    """
    Load drilling data from Excel file with robust data cleaning
    """
    if not file_path:
        raise ValueError("No file path provided. Please specify the path to your Excel file.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not (file_path.endswith('.xlsx') or file_path.endswith('.xls')):
        raise ValueError("Unsupported file format. Please use .xlsx or .xls files")
    
    try:
        print(f"Loading data from: {file_path}")
        # Load with header in row 2 (0-indexed), skip the first 2 rows
        data = pd.read_excel(file_path, header=2)
        print(f"Excel file loaded successfully. Shape: {data.shape}")
        print(f"Available columns: {list(data.columns)}")
        
        # Create column mapping based on your data structure
        column_mapping = {
            'Depth': 'Depth',
            'Mob': 'WOB',  # Assuming Mob is Weight on Bit
            'Rop': 'ROP',  # Rate of Penetration
            'Rpm': 'RPM',  # Revolutions per minute
            'Wob': 'WOB_alt',  # Alternative WOB column
            'Torque': 'Surface_Torque',
            'Gpm': 'Q',  # Flow rate (Gallons per minute)
            'Spp': 'SPP',  # Standpipe Pressure
            'Mw': 'Hook_Load',  # Using Mud Weight as proxy for Hook Load
            'Vis': 'Viscosity'  # Viscosity
        }
        
        # Rename columns to match expected format
        data_renamed = data.rename(columns=column_mapping)
        
        # Use the appropriate WOB column (prefer 'Wob' over 'Mob' if both exist)
        if 'Wob' in data.columns and 'WOB_alt' in data_renamed.columns:
            data_renamed['WOB'] = clean_numeric_column(data['Wob'])
        elif 'Mob' in data.columns:
            data_renamed['WOB'] = clean_numeric_column(data['Mob'])
        
        # Clean all numeric columns
        numeric_columns = ['WOB', 'ROP', 'RPM', 'Surface_Torque', 'Q', 'SPP', 'Depth', 'Hook_Load', 'Viscosity']
        
        for col in numeric_columns:
            if col in data_renamed.columns:
                print(f"Cleaning column: {col}")
                original_col = None
                # Find the original column name
                for orig, mapped in column_mapping.items():
                    if mapped == col and orig in data.columns:
                        original_col = orig
                        break
                
                if original_col:
                    data_renamed[col] = clean_numeric_column(data[original_col])
                else:
                    data_renamed[col] = clean_numeric_column(data_renamed[col])
                
                # Print some statistics about the cleaning
                non_null_count = data_renamed[col].notna().sum()
                print(f"  {col}: {non_null_count} valid values out of {len(data_renamed)}")
                if non_null_count > 0:
                    print(f"  Range: {data_renamed[col].min():.2f} to {data_renamed[col].max():.2f}")
        
        # Check if we have the required columns after mapping
        required_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load', 'ROP', 'Surface_Torque']
        available_columns = []
        missing_columns = []
        
        for col in required_columns:
            if col in data_renamed.columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"Warning: Missing columns after mapping: {missing_columns}")
            
            # Try to use available columns with some substitutions
            if 'Hook_Load' in missing_columns and 'Viscosity' in data_renamed.columns:
                data_renamed['Hook_Load'] = data_renamed['Viscosity']
                missing_columns.remove('Hook_Load')
                available_columns.append('Hook_Load')
                print("Using 'Viscosity' column as Hook_Load")
        
        # Remove rows with missing values in key columns
        key_columns = [col for col in required_columns if col not in missing_columns]
        print(f"Key columns for analysis: {key_columns}")
        
        # Check for missing values before dropping
        print("\nMissing values per column:")
        for col in key_columns:
            missing_count = data_renamed[col].isna().sum()
            print(f"  {col}: {missing_count} missing values")
        
        # Drop rows with any missing values in key columns
        data_clean = data_renamed[key_columns].dropna()
        
        print(f"\nAfter cleaning:")
        print(f"  Original rows: {len(data_renamed)}")
        print(f"  Clean rows: {len(data_clean)}")
        print(f"  Rows removed: {len(data_renamed) - len(data_clean)}")
        
        if len(data_clean) == 0:
            raise ValueError("No valid data rows remaining after cleaning. Please check your data format.")
        
        # Display sample of cleaned data
        print(f"\nSample of cleaned data:")
        print(data_clean.head())
        
        print(f"\nData types after cleaning:")
        print(data_clean.dtypes)
        
        return data_clean
        
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")

# Rest of the classes remain the same...
class DrillingDataset(Dataset):
    """Custom dataset for drilling data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DrillingMLP(nn.Module):
    """Multi-layer perceptron for drilling parameter prediction"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(DrillingMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())  # Using Tanh as in the paper
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Glorot uniform as mentioned in paper)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

# class DrillingPredictor:
#     """PyTorch-based drilling parameter predictor"""
    
#     def __init__(self, hidden_neurons, learning_rate=0.001, 
#                  batch_size=32, epochs=100, dropout_rate=0.2):
#         self.hidden_neurons = hidden_neurons
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.dropout_rate = dropout_rate
#         self.scaler_X = StandardScaler()
#         self.scaler_y = StandardScaler()
#         self.model = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
    
#     def prepare_data(self, X, y):
#         """Prepare and scale the data"""
#         X_scaled = self.scaler_X.fit_transform(X)
#         y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
#         return X_scaled, y_scaled
    
#     def train(self, X_train, y_train, X_val=None, y_val=None):
#         """Train the neural network"""
        
#         # Prepare data
#         X_train_scaled, y_train_scaled = self.prepare_data(X_train, y_train)
        
#         if X_val is not None and y_val is not None:
#             X_val_scaled = self.scaler_X.transform(X_val)
#             y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
#         # Create datasets and dataloaders
#         train_dataset = DrillingDataset(X_train_scaled, y_train_scaled)
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
#         # Initialize model - single hidden layer as per paper
#         input_size = X_train_scaled.shape[1]
#         self.model = DrillingMLP(input_size, [self.hidden_neurons], 1, self.dropout_rate)
#         self.model.to(self.device)
        
#         # Loss function and optimizer (Adam as mentioned in paper)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
#         # Training loop
#         train_losses = []
#         val_losses = []
        
#         print(f"Starting training with {self.hidden_neurons} hidden neurons...")
#         for epoch in range(self.epochs):
#             self.model.train()
#             epoch_loss = 0.0
            
#             for batch_X, batch_y in train_loader:
#                 batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
#                 optimizer.zero_grad()
#                 outputs = self.model(batch_X).squeeze()
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
            
#             avg_train_loss = epoch_loss / len(train_loader)
#             train_losses.append(avg_train_loss)
            
#             # Validation
#             if X_val is not None and y_val is not None:
#                 self.model.eval()
#                 with torch.no_grad():
#                     X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
#                     y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
#                     val_outputs = self.model(X_val_tensor).squeeze()
#                     val_loss = criterion(val_outputs, y_val_tensor).item()
#                     val_losses.append(val_loss)
                
#                 if (epoch + 1) % 50 == 0:
#                     print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
#             else:
#                 if (epoch + 1) % 50 == 0:
#                     print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.6f}')
        
#         print("Training completed!")
#         return train_losses, val_losses
    
#     def predict(self, X):
#         """Make predictions"""
#         if self.model is None:
#             raise ValueError("Model not trained yet!")
        
#         self.model.eval()
#         X_scaled = self.scaler_X.transform(X)
        
#         with torch.no_grad():
#             X_tensor = torch.FloatTensor(X_scaled).to(self.device)
#             predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
#         # Inverse transform predictions
#         predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
#         return predictions
    
#     def evaluate(self, X_test, y_test):
#         """Evaluate model performance"""
#         predictions = self.predict(X_test)
        
#         mse = mean_squared_error(y_test, predictions)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, predictions)
        
#         # Calculate AARE (Average Absolute Relative Error) as in the paper
#         # Avoid division by zero
#         mask = y_test != 0
#         if np.any(mask):
#             aare = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
#         else:
#             aare = np.inf
        
#         return {
#             'mse': mse,
#             'rmse': rmse,
#             'r2': r2,
#             'aare': aare,
#             'predictions': predictions
#         }

def train_drilling_models(data, rop_params=None, torque_params=None):
    """Train both ROP and Torque models using PyTorch"""

    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values

    # ----- ROP model -----
    print("Training ROP Model...")
    y_rop = data['ROP'].values
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )

    if rop_params is not None:
        rop_params = list(rop_params)  # Ensure plain list
        # Use first value for single-layer model compatibility
        hidden_sizes = [int(rop_params[0])]
        learning_rate = float(rop_params[3])
        batch_size = int(rop_params[4])
        dropout_rate = float(rop_params[2])
    else:
        hidden_sizes = [23]
        learning_rate = 0.001
        batch_size = 32
        dropout_rate = 0.2

    rop_model = DrillingPredictor(
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=200,
        dropout_rate=dropout_rate
    )

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_rop_train, test_size=0.2, random_state=42
    )
    rop_model.train(X_train_split, y_train_split, X_val_split, y_val_split)
    rop_results = rop_model.evaluate(X_test, y_rop_test)

    # ----- Torque model -----
    print("\nTraining Torque Model...")
    y_torque = data['Surface_Torque'].values
    X_train, X_test, y_torque_train, y_torque_test = train_test_split(
        X, y_torque, test_size=0.2, random_state=42
    )

    if torque_params is not None:
        torque_params = list(torque_params)  # Ensure plain list
        hidden_sizes = [int(torque_params[0])]
        learning_rate = float(torque_params[3])
        batch_size = int(torque_params[4])
        dropout_rate = float(torque_params[2])
    else:
        hidden_sizes = [27]
        learning_rate = 0.001
        batch_size = 32
        dropout_rate = 0.2

    torque_model = DrillingPredictor(
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=200,
        dropout_rate=dropout_rate
    )

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_torque_train, test_size=0.2, random_state=42
    )
    torque_model.train(X_train_split, y_train_split, X_val_split, y_val_split)
    torque_results = torque_model.evaluate(X_test, y_torque_test)

    return rop_model, torque_model, rop_results, torque_results

def objective_function(params, rop_model, torque_model, sample_features):
    """
    Multi-objective function to maximize ROP and keep torque within constraints
    params: [WOB, RPM] - the parameters to optimize
    """
    wob, rpm = params
    
    # Create feature vector using sample as template
    features = sample_features.copy()
    
    # Find WOB and RPM indices in the feature array
    feature_names = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    wob_idx = feature_names.index('WOB')
    rpm_idx = feature_names.index('RPM')
    
    # Update WOB and RPM in the feature vector
    features[wob_idx] = wob
    features[rpm_idx] = rpm
    
    # Reshape for prediction
    features_array = features.reshape(1, -1)
    
    # Predict ROP (maximize) and torque (constrain)
    predicted_rop = rop_model.predict(features_array)[0]
    predicted_torque = torque_model.predict(features_array)[0]
    
    # Multi-objective: maximize ROP, penalize if torque outside 13k-19k Lb.Ft
    torque_penalty = 0
    if predicted_torque < 13000 or predicted_torque > 19000:
        torque_penalty = abs(predicted_torque - 16000) * 0.01  # Penalty factor
    
    # Return negative ROP (since DE minimizes) plus penalty
    return -predicted_rop + torque_penalty

def optimize_drilling_parameters(rop_model, torque_model, X_data, feature_columns):
    """
    Use Differential Evolution to find optimal WOB and RPM
    """
    # Get data ranges for bounds
    feature_names = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    available_features = [col for col in feature_names if col in feature_columns]
    
    wob_idx = available_features.index('WOB')
    rpm_idx = available_features.index('RPM')
    
    wob_min, wob_max = X_data[:, wob_idx].min(), X_data[:, wob_idx].max()
    rpm_min, rpm_max = X_data[:, rpm_idx].min(), X_data[:, rpm_idx].max()
    
    bounds = [
        (wob_min, wob_max),    # WOB bounds
        (rpm_min, rpm_max)     # RPM bounds
    ]
    
    print(f"Optimizing within bounds: WOB [{wob_min:.1f}, {wob_max:.1f}], RPM [{rpm_min:.1f}, {rpm_max:.1f}]")
    
    # Use last data point as template for other features
    sample_features = X_data[-1].copy()
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(rop_model, torque_model, sample_features),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        seed=42
    )
    
    optimal_wob, optimal_rpm = result.x
    return optimal_wob, optimal_rpm, -result.fun  # Convert back to positive ROP


if __name__ == "__main__":
    # Specify your Excel file path
    excel_file = r"E:\Data\pure\drill operation\Bit Data#1214#MI#131.xlsx"
    
    try:
        # Load data
        drilling_data = load_drilling_data(excel_file)
        
        # Find best ANN hyperparameters for ROP
        best_params_rop = optimize_ann_hyperparameters(drilling_data, target_column='ROP')

        # Find best ANN hyperparameters for Torque if needed
        best_params_torque = optimize_ann_hyperparameters(drilling_data, target_column='Surface_Torque')

        # Train models using optimized params
        rop_model, torque_model, rop_results, torque_results = train_drilling_models(
            drilling_data,
            rop_params=best_params_rop,
            torque_params=best_params_torque
        )
        
        print("\n" + "="*50)
        print("MODELS TRAINED SUCCESSFULLY!")
        print("="*50)
        print(f"ROP Model Performance:")
        print(f"  RMSE: {rop_results['rmse']:.4f}")
        print(f"  R²: {rop_results['r2']:.4f}")
        print(f"  AARE: {rop_results['aare']:.2f}%")
        print(f"\nTorque Model Performance:")
        print(f"  RMSE: {torque_results['rmse']:.4f}")
        print(f"  R²: {torque_results['r2']:.4f}")
        print(f"  AARE: {torque_results['aare']:.2f}%")
        
        # =============================================================================
        # OPTIMIZATION SECTION - Find Optimal Drilling Parameters
        # =============================================================================
        print("\n" + "="*50)
        print("DIFFERENTIAL EVOLUTION OPTIMIZATION")
        print("="*50)
        
        try:
            # Prepare data for optimization
            feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
            available_features = [col for col in feature_columns if col in drilling_data.columns]
            X = drilling_data[available_features].values
            
            # Run optimization to find optimal WOB and RPM
            optimal_wob, optimal_rpm, expected_rop = optimize_drilling_parameters(
                rop_model, torque_model, X, available_features
            )
            
            print(f"\nOptimal Parameters Found:")
            print(f"- Optimal WOB: {optimal_wob:.2f}")
            print(f"- Optimal RPM: {optimal_rpm:.2f}")
            print(f"- Expected ROP: {expected_rop:.2f}")
            
            # Verify the torque constraint
            sample_features = X[-1].copy()
            wob_idx = available_features.index('WOB')
            rpm_idx = available_features.index('RPM')
            sample_features[wob_idx] = optimal_wob
            sample_features[rpm_idx] = optimal_rpm
            
            predicted_torque = torque_model.predict(sample_features.reshape(1, -1))[0]
            
            print(f"- Predicted Torque: {predicted_torque:.0f} Lb.Ft")
            print(f"- Torque within constraints (13k-19k): {'✓' if 13000 <= predicted_torque <= 19000 else '✗'}")
            
            # Compare with current parameters
            current_wob = sample_features[wob_idx] if 'optimal_wob' not in locals() else X[-1, wob_idx]
            current_rpm = sample_features[rpm_idx] if 'optimal_rpm' not in locals() else X[-1, rpm_idx]
            current_rop = rop_model.predict(X[-1].reshape(1, -1))[0]
            current_torque = torque_model.predict(X[-1].reshape(1, -1))[0]
            
            print(f"\nComparison with current parameters:")
            print(f"- Current WOB: {X[-1, wob_idx]:.2f} → Optimal: {optimal_wob:.2f} (Δ: {optimal_wob - X[-1, wob_idx]:+.2f})")
            print(f"- Current RPM: {X[-1, rpm_idx]:.2f} → Optimal: {optimal_rpm:.2f} (Δ: {optimal_rpm - X[-1, rpm_idx]:+.2f})")
            print(f"- Current ROP: {current_rop:.2f} → Expected: {expected_rop:.2f} (Δ: {expected_rop - current_rop:+.2f})")
            print(f"- Current Torque: {current_torque:.0f} → Expected: {predicted_torque:.0f} (Δ: {predicted_torque - current_torque:+.0f})")
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
