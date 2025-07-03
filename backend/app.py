from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image, ImageOps
import onnxruntime as ort
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Load ONNX model
ort_session = ort.InferenceSession("saved_model/model.onnx")

def preprocess_image(image):
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and scale to [0, 1]
    arr = np.array(image).astype(np.float32) / 255.0
    # Normalize using MNIST mean and std
    arr = (arr - 0.1307) / 0.3081
    # Add channel and batch dimensions: [1, 1, 28, 28]
    arr = arr[np.newaxis, np.newaxis, :, :]
    return arr

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and convert to grayscale
    image = Image.open(io.BytesIO(await file.read())).convert("L")

    # Invert colors manually (canvas = black on white, MNIST = white on black)
    image = ImageOps.invert(image)

    # Preprocess the image
    input_array = preprocess_image(image)

    # Convert the preprocessed array back to a PNG and save for inspection
    # Undo normalization for visualization
    arr_vis = input_array[0, 0] * 0.3081 + 0.1307  # de-normalize
    arr_vis = np.clip(arr_vis * 255, 0, 255).astype(np.uint8)
    img_vis = Image.fromarray(arr_vis)
    img_vis.save("preprocessed.png")  # This will save the image on the server

    # ONNX inference
    ort_inputs = {"input": input_array.astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs[0], axis=1)[0]

    return {"prediction": int(pred)}

