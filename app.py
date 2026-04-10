from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import requests

from model import SimpleCNN

app = FastAPI()

# -------------------------------
# MODEL SETUP
# -------------------------------

MODEL_PATH = "saved_model/model.pth"

def download_model(url, path):
    print("Downloading model...")
    r = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete!")

# 👉 PUT YOUR GOOGLE DRIVE DIRECT LINK HERE
MODEL_URL = "PASTE_YOUR_MODEL_LINK_HERE"

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_model", exist_ok=True)
    try:
        download_model(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print("❌ Error downloading model:", e)

# Load model safely
model = SimpleCNN()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ ERROR loading model:", e)

# -------------------------------
# IMAGE TRANSFORM
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# -------------------------------
# ROUTES
# -------------------------------

@app.get("/")
def home():
    return {"message": "CNN API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return {"prediction": classes[predicted.item()]}

    except Exception as e:
        return {"error": str(e)}
