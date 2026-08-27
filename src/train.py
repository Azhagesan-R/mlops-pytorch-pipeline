import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from model import get_model
from dataset import get_dataloaders

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
def log_json_line(metrics: dict):
    print(json.dumps(metrics), flush=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total

def main():
    config_path = Path(os.getenv("CONFIG_PATH", "configs/training_config.yaml"))
    config = load_config(str(config_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_json_line({"status": "initializing", "device": str(device)})

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    
    data_dir = config["data"]["data_dir"]
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config["training"]["batch_size"],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    
    checkpoint_dir = Path(os.getenv("MODEL_OUTPUT_DIR", config["output"]["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["output"]["model_name"]

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
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        log_json_line(log_entry)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / model_name
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_json_line({"event": "early_stopping", "epoch": epoch + 1})
                break
                
    log_json_line({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})

if __name__ == "__main__":
    main()
