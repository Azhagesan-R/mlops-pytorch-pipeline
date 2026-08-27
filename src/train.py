import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloader

def load_config(config_path: str) -> dict:
    """Loads the training configuration YAML file securely with a fallback check."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing configuration YAML file: {e}", file=sys.stderr)
        sys.exit(1)

def train():
    # 1. Pipeline path resolution using environmental variable definitions
    config_path = os.getenv("CONFIG_PATH", "configs/training_config.yaml")
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "/app/checkpoints")
    
    print(f"Initializing pipeline with Config: {config_path}")
    config = load_config(config_path)

    # 2. Compute execution layer detection 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training runtime environment configured on: {device}")

    # 3. Model setup using config dimensions
    model = SimpleCNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))

    # 4. Pipeline ingestion layer
    print("Assembling PyTorch DataLoaders...")
    dataloader = get_dataloader(
        batch_size=config['training']['batch_size'], 
        num_samples=config['training']['num_samples']
    )

    # 5. Core optimization training loop
    print(f"Starting training lifecycle for {config['training']['epochs']} epochs...")
    model.train()
    for epoch in range(config['training']['epochs']):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1:02d}/{config['training']['epochs']:02d}] Batch-Averaged Cross-Entropy Loss: {epoch_loss:.4f}")

    # 6. Structured artifact serialization
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pth")
    
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Success: Optimization complete. Serialized weight matrices saved to persistent volume check at: {model_path}")
    except Exception as e:
        print(f"Critical Error: Failed to save model state dictionary: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train()
