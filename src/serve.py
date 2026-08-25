import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from torchvision import transforms
from model import SimpleCNN

app = FastAPI(title="PyTorch Serving Service", version="1.0")

# Initialize and load model global reference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model_path = os.getenv("MODEL_PATH", "/mnt/models/model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"Successfully loaded weight weights from {model_path}")
else:
    print(f"Warning: weights file missing at {model_path}. Serving random initializations.")
    model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.get("/healthz/live", status_code=200)
def liveness_probe():
    return {"status": "alive"}

@app.get("/healthz/ready", status_code=200)
def readiness_probe():
    # Ready if model state is accessible on context device
    if model is not None:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model uninitialized")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid media type file")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        return {"class_id": int(predicted.item())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
