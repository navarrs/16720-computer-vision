import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import string

from torchvision import transforms, datasets


#
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
batch_size = 32
lr_rate = 1e-5
n_classes = 17
n_epochs = 30
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#
# DATASET
# ------------------------------------------------------------------------------

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = datasets.ImageFolder(root="../data/oxford-flowers17/train",
                                     transform=data_transform)
val_dataset = datasets.ImageFolder(root="../data/oxford-flowers17/val",
                                   transform=data_transform)
test_dataset = datasets.ImageFolder(root="../data/oxford-flowers17/test",
                                    transform=data_transform)

data_loaders = {
    "train": torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True),
    "train_size": len(train_dataset),
    "val": torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False),
    "val_size": len(val_dataset),
    "test": torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False),
    "test_size": len(test_dataset)
}

#
# Network
# ------------------------------------------------------------------------------


class CNN(nn.Module):
    def __init__(self, channels=1, n_classes=17):
        super(CNN, self).__init__()
        self.convs = nn.Sequential(
            # Layer 1
            nn.Conv2d(channels, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            # Layer 4
            nn.Linear(120, 84),
            nn.Tanh(),
            # Layer 5
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, F.softmax(x, dim=1)

#
# Visualization
# ------------------------------------------------------------------------------


def plot(losses, accs, n_epochs, lr_rate, batch_size, net_type=''):
    ep = np.arange(start=0, stop=n_epochs, step=1)

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle(
        f"iters:{n_epochs} - batch-size: {batch_size} - lr: {lr_rate}",
        fontsize=12)
    axs[0].plot(ep, losses["train"], 'k.-', label='train loss')
    axs[0].plot(ep, losses["val"], 'g.-', label='val loss')
    axs[0].set_title('Loss vs Epochs')
    axs[0].set_xlabel('epochs ')
    axs[0].set_ylabel('loss')
    axs[0].legend(loc='upper right', borderaxespad=0.)

    axs[1].plot(ep, accs["train"], 'k.-', label='train acc')
    axs[1].plot(ep, accs["val"], 'g.-', label='valid acc')
    axs[1].set_title('Accuracy vs Epochs')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].legend(loc='upper left', borderaxespad=0.)

    plt.savefig("../out/q6/{}_loss-{:.3f}_acc-{:.3f}_iter-{}_lr-{}_batch-{}.png"
                .format(net_type, losses["val"][-1], accs["val"][-1],
                        n_epochs, lr_rate, batch_size))

    plt.show()
    # plt.close()

#
# Train
# ------------------------------------------------------------------------------


def train():
    net = CNN(channels=3, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    losses = {"train": [], "val": []}
    accs = {"train": [], "val": []}

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        for phase in ["train", "val"]:
            running_loss, running_acc = 0.0, 0.0

            for i, data in enumerate(data_loaders[phase], 0):
                X, y = data
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_hat, y_p = net(X)

                    loss = criterion(y_p, y)
                    _, pred = torch.max(y_p.data, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_acc += (pred == y.data).sum()

            N = data_loaders[phase + "_size"]
            losses[phase].append(running_loss / N)
            accs[phase].append(running_acc / N)
            print("\t{}:\t loss: {:.3f}\t acc: {:.3f}"
                  .format(phase, losses[phase][epoch], accs[phase][epoch]))

    # Plot loss and accuracy
    plot(losses, accs, n_epochs, lr_rate, batch_size, net_type='scratch')


train()
