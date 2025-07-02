import torch
from torch import nn, optim
from torchvision import datasets, transforms 
import wandb

from model.model import DigitCNN

wandb.init(project="digit-recognizer")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=False
)

model = DigitCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(20):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
    wandb.log({"train_loss": loss.item()})

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item() * x.size(0)
            _, predicted = pred.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_test_loss = test_loss / total
    accuracy = correct / total
    print(f"Test loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")
    wandb.log({"test_loss": avg_test_loss, "test_accuracy": accuracy})

# Save model
torch.save(model.state_dict(), "./saved_model/model.pth")
