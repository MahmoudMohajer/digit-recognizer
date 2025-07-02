from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from model.model import DigitCNN

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("saved_model/model.pth", map_location="cpu"))
model.eval()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Inside /predict
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = ImageOps.invert(image)  # ✅ Invert colors
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    return {"prediction": pred}
