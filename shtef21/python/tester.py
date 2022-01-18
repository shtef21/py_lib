
import train as shtef

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader as DL
from torchvision import datasets 
from torchvision import transforms as T

class Flatten(nn.Module):
    def forward(_, x): return x.view(x.shape[0], -1)

model = nn.Sequential(
    Flatten(),
    nn.Linear(28 * 28, 10)
)

def test():
    train_loader = DL(
        datasets.MNIST('', train=False, download=True, transform=T.ToTensor()),
        batch_size=16,
    )
    correct, total = 0, 0
    for images, labels in train_loader:
        out = model(images)
        preds = torch.argmax(out, -1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total

print(f"""
    acc 1: {test()}
    acc 2: {test()}
""")

shtef.train(
    model,
    epochs=1,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.SGD(model.parameters(), 1e-3, 0.9),
    trainset=datasets.MNIST('', transform=T.ToTensor(), download=True),
)

print(f"""
    acc 1: {test()}
    acc 2: {test()}
""")

