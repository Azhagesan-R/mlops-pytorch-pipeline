import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloader

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def log_json_line(metrics: dict):
    print(json.dumps(metrics), flush=True)

def train():
    config_path = os.getenv("CONFIG_PATH", "configs/training_config.yaml")
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_json_line({"status": "initializing", "device": str(device)})

    model = SimpleCNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))

    train_loader = get_dataloader(batch_size=config['training']['batch_size'], train=True)
    val_loader = get_dataloader(batch_size=config['training']['batch_size'], train=False)

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = config['training'].get('patience', 3)
    max_epochs = config['training']['epochs']

    for epoch in range(max_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Structured Metrics Output as requested (JSON lines format)
        log_json_line({
            "epoch": epoch + 1,
            "train_loss": round(epoch_train_loss, 4),
            "train_accuracy": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_accuracy": round(epoch_val_acc, 4)
        })

        # Early Stopping Logic 
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            epochs_no_improve = 0
            output_dir = os.getenv("MODEL_OUTPUT_DIR", "./models")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_json_line({"status": "early_stopping_triggered", "stopped_at_epoch": epoch + 1})
                break

if __name__ == "__main__":
    train()
