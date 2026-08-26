import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from torchvision import transforms
from model import SimpleCNN

app = FastAPI(title="PyTorch Inference Engine", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model_path = os.getenv("MODEL_PATH", "/mnt/models/model.pth")

# Load model weights on initialization if present
model_loaded = False
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        model_loaded = True
        print(f"Successfully mounted weights target from: {model_path}")
    except Exception as e:
        print(f"Error reading model check file structural parameters: {str(e)}")
else:
    print(f"Warning: Checkpoint artifact missing at {model_path}. Serving default tensor shapes.")
    model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.get("/health", status_code=200)
def health_check():
    """Returns 200 configuration response if model parameters map correctly onto processing memory."""
    if model is not None:
        return {"status": "healthy", "model_loaded": model_loaded}
    raise HTTPException(status_code=503, detail="Inference application unit uninitialized")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid transmission file media payload extension format type.")
    
    try:
        payload = await file.read()
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)
            
        return {
            "predicted_class": int(torch.argmax(probabilities).item()),
            "probabilities": [round(float(p), 5) for p in probabilities.tolist()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
