from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import torchvision.transforms as T
from model.model import DigitCNN
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Just before prediction


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
    T.Resize((28, 28)),
    ImageOps.invert,  # We'll apply manually below
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L")  # Grayscale

    # 🔄 Invert color: your canvas is black on white, MNIST is white on black
    image = ImageOps.invert(image)

    # ✂️ Resize and normalize
    image = transform(image).unsqueeze(0)  # Shape: (1, 1, 28, 28)


    # Just before prediction
    TF.to_pil_image(image.squeeze()).save("debug_input.png")

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    return {"prediction": pred}


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1, shuffle=True
)

x, y = next(iter(test_loader))
pred = model(x).argmax(dim=1)
print(f"Label: {y.item()}, Prediction: {pred.item()}")