from torchvision import datasets, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from model.model import DigitCNN
import torchvision.transforms.functional as TF

app = FastAPI()

# CORS
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

# Define transform (no inversion here)
transform = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and convert to grayscale
    image = Image.open(io.BytesIO(await file.read())).convert("L")

    # ✅ Invert colors manually (canvas = black on white, MNIST = white on black)
    image = ImageOps.invert(image)

    # ✅ Apply transform
    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 28, 28]

    # ✅ Save debug image (non-normalized for visual check)
    debug_img = TF.to_pil_image(image_tensor.squeeze(0) * 0.3081 + 0.1307)  # unnormalize
    debug_img.save("debug_input.png")

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
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