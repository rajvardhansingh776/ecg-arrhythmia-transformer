import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ECGEncoder(nn.Module):
    """Simple ECG encoder for self-supervised learning"""
    def __init__(self, input_dim=500, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

def contrastive_loss(z1, z2, temperature=0.07):
    """Simplified NT-Xent loss"""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    batch_size = z1.shape[0]
    
    # Concatenate representations
    z = torch.cat([z1, z2], dim=0)
    
    # Similarity matrix
    sim = torch.matmul(z, z.T) / temperature
    
    # Labels for positive pairs
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels, labels])
    
    # Simple contrastive loss
    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss

def jitter(x, std=0.001):
    """Add small random noise"""
    return x + np.random.normal(0, std, x.shape)

def scaling(x, factor=0.9):
    """Scale signal"""
    return x * factor

def load_data(dataset_name="ptbxl"):
    """Load preprocessed data"""
    try:
        X = np.load(f"data/processed/{dataset_name}_signals.npy")
        print(f"[OK] Loaded {dataset_name}: X shape {X.shape}")
        return X
    except FileNotFoundError:
        print(f"[WARN] {dataset_name} data not found, creating synthetic data")
        num_samples = 500
        seq_len = 500
        X = np.random.randn(num_samples, seq_len).astype(np.float32)
        return X

def pretrain_ssl(dataset_name="ptbxl", epochs=10, batch_size=32, lr=1e-3):
    """Self-supervised pretraining"""
    
    print("\n" + "="*50)
    print(f"SELF-SUPERVISED PRETRAINING ON {dataset_name.upper()}")
    print("="*50 + "\n")
    
    # Load data
    X = load_data(dataset_name)
    
    # Create dataset
    X_tensor = torch.tensor(X).float()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ECGEncoder(input_dim=X.shape[1])
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device)
            
            # Create augmented views
            x1_np = jitter(x.cpu().numpy())
            x2_np = scaling(x.cpu().numpy())
            
            x1 = torch.tensor(x1_np).float().to(device)
            x2 = torch.tensor(x2_np).float().to(device)
            
            # Forward pass
            z1 = model(x1)
            z2 = model(x2)
            
            # Compute loss
            loss = contrastive_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - SSL Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ssl_encoder_{dataset_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] SSL model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    try:
        model = pretrain_ssl(dataset_name="ptbxl", epochs=5, batch_size=32)
        print("\n[OK] SSL pretraining complete!")
    except Exception as e:
        print(f"\n[ERROR] Error during SSL pretraining: {e}")
        import traceback
        traceback.print_exc()