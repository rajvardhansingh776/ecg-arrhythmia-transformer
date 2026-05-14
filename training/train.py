import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ECGTransformer(nn.Module):
    def __init__(self, seq_len, classes, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=256
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, classes)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

def load_data(dataset_name="ptbxl"):
    """Load preprocessed data"""
    try:
        X = np.load(f"data/processed/{dataset_name}_signals.npy")
        y = np.load(f"data/processed/{dataset_name}_labels.npy")
        print(f"[OK] Loaded {dataset_name}: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except FileNotFoundError:
        print(f"[WARN] {dataset_name} data not found, creating synthetic data")
        num_samples = 500
        seq_len = 500
        num_classes = 5
        X = np.random.randn(num_samples, seq_len).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        return X, y

def train_model(dataset_name="ptbxl", epochs=20, batch_size=32, lr=1e-4):
    """Train ECG transformer model"""
    
    print("\n" + "="*50)
    print(f"TRAINING ON {dataset_name.upper()}")
    print("="*50 + "\n")
    
    # Load data
    X, y = load_data(dataset_name)
    
    # Create dataset
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).long()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ECGTransformer(seq_len=X.shape[1], classes=len(np.unique(y)))
    model = model.to(device)
    
    # Compute class weights
    try:
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        weights = torch.tensor(weights).float().to(device)
    except:
        weights = None
    
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{dataset_name}_transformer.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    try:
        model = train_model(dataset_name="ptbxl", epochs=10, batch_size=32)
        print("\n[OK] Training complete!")
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()