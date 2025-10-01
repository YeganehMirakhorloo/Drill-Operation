import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
    
    def __init__(self, hidden_sizes=[64, 32, 16], learning_rate=0.001, 
                 batch_size=32, epochs=100, dropout_rate=0.2):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
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
        """Train the neural network"""
        
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
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print("Starting training...")
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
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }

def train_drilling_models(data):
    """Train both ROP and Torque models using PyTorch"""
    
    # Prepare features and targets
    feature_columns = ['WOB', 'RPM', 'SPP', 'Q', 'Depth', 'Hook_Load']
    X = data[feature_columns].values
    
    # Train ROP model
    print("Training ROP Model...")
    y_rop = data['ROP'].values
    X_train, X_test, y_rop_train, y_rop_test = train_test_split(
        X, y_rop, test_size=0.2, random_state=42
    )
    
    rop_model = DrillingPredictor(
        hidden_sizes=[23, 16, 8],  # Based on the paper's optimal 23 neurons
        learning_rate=0.001,
        batch_size=32,
        epochs=200
    )
    
    # Split training data for validation
    X_train_split, X_val_split, y_rop_train_split, y_rop_val_split = train_test_split(
        X_train, y_rop_train, test_size=0.2, random_state=42
    )
    
    rop_model.train(X_train_split, y_rop_train_split, X_val_split, y_rop_val_split)
    rop_results = rop_model.evaluate(X_test, y_rop_test)
    
    print(f"ROP Model - RMSE: {rop_results['rmse']:.4f}, R²: {rop_results['r2']:.4f}")
    
    # Train Torque model
    print("\nTraining Torque Model...")
    y_torque = data['Surface_Torque'].values
    X_train, X_test, y_torque_train, y_torque_test = train_test_split(
        X, y_torque, test_size=0.2, random_state=42
    )
    
    torque_model = DrillingPredictor(
        hidden_sizes=[27, 18, 9],  # Based on the paper's optimal 27 neurons
        learning_rate=0.001,
        batch_size=32,
        epochs=200
    )
    
    X_train_split, X_val_split, y_torque_train_split, y_torque_val_split = train_test_split(
        X_train, y_torque_train, test_size=0.2, random_data=42
    )
    
    torque_model.train(X_train_split, y_torque_train_split, X_val_split, y_torque_val_split)
    torque_results = torque_model.evaluate(X_test, y_torque_test)
    
    print(f"Torque Model - RMSE: {torque_results['rmse']:.4f}, R²: {torque_results['r2']:.4f}")
    
    return rop_model, torque_model, rop_results, torque_results
