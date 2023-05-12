
import train as shtef
from torch import nn
from torch import optim
from torchvision import datasets 
from torchvision import transforms as T

class Flatten(nn.Module):
    def forward(_, x): return x.view(x.shape[0], -1)

model = nn.Sequential(
    Flatten(),
    nn.Linear(28 * 28, 10)
)

shtef.train(
    model,
    epochs=3,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.SGD(model.parameters(), 1e-3, 0.9),
    trainset=datasets.MNIST('', transform=T.ToTensor(), download=True),
    testset=datasets.MNIST('', train=False, transform=T.ToTensor(), download=True),
    test_once=True
)
