import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloader

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    config_path = os.getenv("CONFIG_PATH", "configs/training_config.yaml")
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")

    model = SimpleCNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))

    dataloader = get_dataloader(
        batch_size=config['training']['batch_size'], 
        num_samples=config['training']['num_samples']
    )

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
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Loss: {running_loss/len(dataloader):.4f}")

    # Output artifact management via environment variables
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "./models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train()
