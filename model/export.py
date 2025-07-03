import torch
from model.model import DigitCNN

model = DigitCNN()
model.load_state_dict(torch.load("./saved_model/model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, 
    dummy_input, 
    "./saved_model/model.onnx", 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)