import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import string

import torchvision
import torchvision.transforms as transforms

#
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
letters = np.array([_ for _ in string.ascii_uppercase[:26]]
                   + [str(_) for _ in range(10)])

#
# DATASET
# ------------------------------------------------------------------------------
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        labels = np.argmax(labels, axis=1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        rdata = np.asarray([d.reshape(1, 32, 32) for d in data])
        self.data = torch.from_numpy(rdata).float()
        labels = np.argmax(labels, axis=1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#
# Networks
# ------------------------------------------------------------------------------
# For Q6.1.1


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(32, 36)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

# For Q6.1.2


class CNN2(nn.Module):
    def __init__(self, channels=1, n_classes=10):
        super(CNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.drop = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 64 * 8 * 8)
        # x = self.drop(x)
        x = self.fc1(x)
        return self.fc2(x)


class CNN(nn.Module):
    def __init__(self, channels=1, n_classes=10, p=0.5):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.drop = nn.Dropout(p)
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 16 * 5 * 5)
        # x = self.drop(x)
        return self.fc(x)


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

    # plt.savefig("../out/q6/{}_batch-{}_iter-{}_lr-{}_loss-{:.3f}_acc-{:.3f}.png"
    #             .format(net_type, batch_size, n_epochs, lr_rate,
    #                     losses["val"][-1], accs["val"][-1]))

    plt.show()
    plt.close()


def train_cnn(net, data_loaders, dataset, batch_size=72, n_epochs=10, lr_rate=1e-3):
    print("Configuration -- epochs:{}, lr:{}, batch_size:{}"
          .format(n_epochs, lr_rate, batch_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    losses = {"train": [], "val": []}
    accs = {"train": [], "val": []}

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        for phase in ["train", "val"]:
            running_loss, running_acc = 0.0, 0.0

            if phase == "train":
                net.train()
            else:
                net.eval()

            for i, data in enumerate(data_loaders[phase], 0):
                X, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_p = net(X)
                    _, pred = torch.max(y_p.data, 1)
                    loss = criterion(y_p, y)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss * X.size(0)
                running_acc += (pred == y.data).sum()

            N = data_loaders[phase + "_size"]
            losses[phase].append(running_loss / N)
            accs[phase].append(running_acc / N)
            print("\t{}:\t loss: {:.3f}\t acc: {:.3f}"
                  .format(phase, losses[phase][epoch], accs[phase][epoch]))

    # Plot loss and accuracy
    plot(losses, accs, n_epochs, lr_rate,
         batch_size, net_type=f"cnn_{dataset}")

#
# Q6.1.1 Fully connected network
# ------------------------------------------------------------------------------


def q6_1_1(n_epochs=30, lr_rate=1e-3, batch_size=54):
    print("Configuration -- epochs:{}, lr:{}, batch_size:{}"
          .format(n_epochs, lr_rate, batch_size))
    # Load dataset
    train_dataset = Dataset(train_x, train_y)
    val_dataset = Dataset(valid_x, valid_y)

    data_loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True),
        "train_size": len(train_dataset),
        "val": torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False),
        "val_size": len(val_dataset)
    }
    net = FC()
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
                    y_p = net(X)
                    _, pred = torch.max(y_p.data, 1)
                    loss = criterion(y_p, y)

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
    plot(losses, accs, n_epochs, lr_rate, batch_size, net_type='fcn_nist')

#
# Q6.1.2 - CNN NIST
# ------------------------------------------------------------------------------


def q6_1_2():
    train_dataset = Dataset2(train_x, train_y)
    val_dataset = Dataset2(valid_x, valid_y)
    batch_size = 108
    data_loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True),
        "train_size": len(train_dataset),
        "val": torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False),
        "val_size": len(val_dataset)
    }
    net = CNN(channels=1, n_classes=36)
    train_cnn(net, data_loaders, 'nist', batch_size)

#
# Q6.1.3 - CNN CIFAR
# ------------------------------------------------------------------------------


def q6_1_3():
    batch_sizes = [108]  # , 72, 54, 32, 16]
    lr_rates = [1e-3]  # , 3e-2]
    n_epochs = [10]  # , 15, 30]

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    val_dataset = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)

    for lr_rate in lr_rates:
        for epochs in n_epochs:
            for batch_size in batch_sizes:
                data_loaders = {
                    "train": torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True),
                    "train_size": len(train_dataset),
                    "val": torch.utils.data.DataLoader(
                        val_dataset, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True),
                    "val_size": len(val_dataset)
                }
                net = CNN2(channels=3, n_classes=10)
                train_cnn(net, data_loaders, 'cifar',
                          batch_size, epochs, lr_rate)


#
# Q6.1.4 - CNN SUN
# ------------------------------------------------------------------------------

def q6_1_4():
    batch_sizes = [50]  # , 32, 74]
    lr_rates = [1e-3]  # 1e-3, 1e-2]
    n_classes = 10
    n_epochs = [15]  # , 10]#, 20]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    trans = transforms.Compose([
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = torchvision.datasets.ImageFolder(
        root="../data/sun", transform=trans)
    N = len(dataset)
    val_split = int(0.20 * N)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [N - val_split, val_split])
    print(f"train: {len(train_dataset)} val: {len(val_dataset)}")

    for epochs in n_epochs:
        for lr in lr_rates:
            for batch_size in batch_sizes:
                data_loaders = {
                    "train": torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True),
                    "train_size": len(train_dataset),
                    "val": torch.utils.data.DataLoader(
                        val_dataset, batch_size=1,
                        shuffle=False, num_workers=4, pin_memory=True),
                    "val_size": len(val_dataset)
                }
                net = CNN2(channels=3, n_classes=n_classes)
                train_cnn(net, data_loaders, 'sun', batch_size, epochs, lr)

                # Alexnet
                # net = torchvision.models.alexnet()
                # train_cnn(net, data_loaders, 'sun', batch_size, epochs, lr)


if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Q6.1.1 - FC  - NIST
    print(f"******************************************************************")
    print(f"Q 6.1.1 Fully Connected Network - NIST")
    q6_1_1()

    # --------------------------------------------------------------------------
    # Q6.1.2 - CNN - NIST
    print(f"******************************************************************")
    print(f"Q 6.1.2 ConvNet - NIST")
    q6_1_2()

    # --------------------------------------------------------------------------
    # Q6.1.3 - CNN - CIFAR
    print(f"******************************************************************")
    print(f"Q 6.1.2 ConvNet - CIFAR10")
    q6_1_3()

    # --------------------------------------------------------------------------
    # Q6.1.4 - CNN - SUN
    print(f"******************************************************************")
    print(f"Q 6.1.2 ConvNet - SUN")
    q6_1_4()
