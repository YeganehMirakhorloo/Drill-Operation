import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_augmentation import augment_training_data

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
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(DrillingMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DrillingPredictor:
    """PyTorch-based drilling parameter predictor"""
    
    def __init__(self, hidden_sizes=[8], learning_rate=0.001, 
                 batch_size=4, epochs=100, dropout_rate=0.3, weight_decay=1e-4):
        self.hidden_sizes = hidden_sizes
        # Restrict learning rate to specific values
        if learning_rate not in [0.001, 0.0005, 0.0001]:
            # Find closest valid value
            valid_lrs = [0.001, 0.0005, 0.0001]
            learning_rate = min(valid_lrs, key=lambda x: abs(x - learning_rate))
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the neural network with early stopping"""
        
        # Prepare data
        X_train_scaled, y_train_scaled = self.prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Create datasets and dataloaders
        train_dataset = DrillingDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train_scaled.shape[1]
        self.model = DrillingMLP(input_size, self.hidden_sizes, 1, self.dropout_rate)
        self.model.to(self.device)
        
        # Loss function and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                              weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"Starting training with LR={self.learning_rate}, Hidden neurons={self.hidden_sizes}...")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                    scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}')
        
        print("Training completed!")
        return train_losses, val_losses
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Calculate AARE
        mask = y_test != 0
        if np.any(mask):
            aare = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
        else:
            aare = np.inf

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'aare': aare,
            'predictions': predictions
        }


def train_drilling_models(data):
    """Train ROP model only using PyTorch"""
    
    # Prepare features and targets
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    
    # Train ROP model with single hidden layer
    print("="*80)
    print("TRAINING ROP MODEL ONLY")
    print("="*80)
    y_rop = data['ROP'].values
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )
    
    # AUGMENT TRAINING DATA FIRST (before validation split)
    print("\nAugmenting ROP training data...")
    X_train_aug, y_rop_train_aug = augment_training_data(
        X_train, y_rop_train, 
        noise_levels=[0.05, 0.10, 0.20],
        verbose=True
    )
    
    rop_model = DrillingPredictor(
        hidden_sizes=[8],  # Single hidden layer
        learning_rate=0.001,  # Must be 0.001, 0.0005, or 0.0001
        batch_size=4,
        epochs=200,
        dropout_rate=0.3,
        weight_decay=1e-4
    )
    
    # Split AUGMENTED training data for validation
    X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
        X_train_aug, y_rop_train_aug, test_size=0.2, random_state=42
    )
    
    rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
    
    # Evaluate on ORIGINAL (non-augmented) train and test sets
    rop_train_results = rop_model.evaluate(X_train, y_rop_train)
    rop_test_results = rop_model.evaluate(X_test, y_rop_test)
    
    print(f"\n{'='*80}")
    print(f"ROP MODEL RESULTS")
    print(f"{'='*80}")
    print(f"Train Set - RMSE: {rop_train_results['rmse']:.4f}, R²: {rop_train_results['r2']:.4f}, AARE: {rop_train_results['aare']:.2f}%")
    print(f"Test Set  - RMSE: {rop_test_results['rmse']:.4f}, R²: {rop_test_results['r2']:.4f}, AARE: {rop_test_results['aare']:.2f}%")
    print(f"{'='*80}\n")
    
    # Return only ROP model and results
    return rop_model, rop_train_results, rop_test_results, X_train, X_test, y_rop_train, y_rop_test
