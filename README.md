# Digit Recognizer

A full-stack digit recognition project using PyTorch, FastAPI, and a simple HTML/JS frontend. Draw a digit in your browser and get real-time predictions powered by a trained CNN on MNIST.

## Features

- **Deep Learning**: Convolutional Neural Network (CNN) built with PyTorch.
- **Training & Logging**: Model training with live metrics logged to [Weights & Biases](https://wandb.ai/).
- **REST API**: FastAPI backend for serving predictions.
- **Frontend**: HTML5 canvas for drawing digits, with instant prediction results.
- **Docker-ready**: Easily containerize backend and frontend (optional).

---

## Project Structure

```
digit-recognizer/
├── backend/
│   └── app.py           # FastAPI backend for prediction
├── frontend/
│   └── index.html       # Simple HTML/JS frontend
├── model/
│   ├── model.py         # CNN model definition
│   └── train.py         # Training script (logs to wandb)
├── saved_model/
│   └── model.pth        # Trained model weights (after training)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd model
python train.py
```
- Training logs and metrics will be sent to your [wandb](https://wandb.ai/) account.

### 3. Start the Backend

```bash
cd ../backend
uvicorn app:app --reload
```

### 4. Launch the Frontend

Just open `frontend/index.html` in your browser.

---

## Usage

1. Draw a digit (0-9) on the canvas.
2. Click **Predict**.
3. The predicted digit will be displayed below the canvas.

---

## API

- **POST** `/predict/`
  - Accepts: PNG image file (28x28 grayscale, white background, black digit)
  - Returns: `{ "prediction": <digit> }`

---

## Notes

- The backend expects images similar to MNIST: white background, black digit. The frontend handles this automatically.
- Model weights are saved to `saved_model/model.pth` after training.
- For best results, train the model before running the backend.

---

## License

MIT License