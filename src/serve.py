import os
import io
import sys
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

app = FastAPI(title="PyTorch Serving Service", version="1.0")

# 1. Compute and weight matrix file configuration path resolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.getenv("MODEL_PATH", "/app/checkpoints/model.pth")

# Instantiate model structure dynamically using class dimension mapping (assuming standard 10 classes or default)
model = SimpleCNN()

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        print(f"Successfully loaded weight matrices from {model_path}")
    except Exception as e:
        print(f"Critical error loading model weights: {e}", file=sys.stderr)
        model.eval().to(device)
else:
    print(f"Warning: Weights file missing at {model_path}. Serving random initializations.", file=sys.stderr)
    model.eval().to(device)

# 2. Tensor transformation layer matching pipeline requirements
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Core health endpoints matching Dockerfile configurations
@app.get("/health", status_code=200)
def structural_health_check():
    """Consolidated basic health endpoint for Docker engine validation layer."""
    if model is not None:
        return {"status": "healthy", "device": str(device)}
    raise HTTPException(status_code=503, detail="Inference application degraded")

@app.get("/healthz/live", status_code=200)
def liveness_probe():
    return {"status": "alive"}

@app.get("/healthz/ready", status_code=200)
def readiness_probe():
    if model is not None:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model uninitialized")

# 4. Ingestion endpoint altered to explicitly match curl requirement (-F "image=@...")
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Accepts image parameter payload form to execute inference operations."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid media type file layout. Expects an image.")
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        return {"class_id": int(predicted.item())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(e)}")
