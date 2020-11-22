import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import string
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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(32, 36)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

# For Q6.1.2


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, 36)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.drop(x)
        x = self.fc1(x)
        return self.fc2(x)


#
# Q6.1.1 Fully connected network
# ------------------------------------------------------------------------------
def q6_1_1(epochs=30, lr_rate = 1e-3, batch_size = 108):

    train_dataset = Dataset(train_x, train_y)
    val_dataset = Dataset(valid_x, valid_y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False)
    net = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)
    losses = []
    accs = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for i, data in enumerate(train_loader, 0):
            X, y = data

            optimizer.zero_grad()

            yout = net(X)
            _, pred = torch.max(yout.data, 1)
            total_acc += (pred == y).sum().item() / y.size(0)

            loss = criterion(yout, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
        accs.append(total_acc)
        if epoch % 2 == 0:    # print every 2000 mini-batches
            print('[epoch %d] loss: %.3f acc: %.3f' %
                  (epoch + 1, total_loss, total_acc))

    ep = np.arange(start=0, stop=epochs, step=1)
    plt.subplot(1, 2, 1)
    plt.plot(ep, losses, 'ko-')
    plt.title('Loss vs Epochs')

    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.subplot(1, 2, 2)
    plt.plot(ep, accs, 'r.-')
    plt.title('Accuracy vs Epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')

    plt.savefig("../out/q6/mlp_loss-{:.3f}_acc-{:.3f}_epocs-{}_lr-{}_batch-{}.png"
                .format(total_loss, total_acc, epochs, lr_rate, batch_size))

    plt.show()

    total = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            X, y = data
            yout = net(X)
            _, pred = torch.max(yout.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    print(f"Validation accuracy: {correct / total}")


def q6_1_2(epochs=10, lr_rate = 1e-3, batch_size = 108):

    train_dataset = Dataset2(train_x, train_y)
    val_dataset = Dataset2(valid_x, valid_y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False)

    net = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)
    losses = []
    accs = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for i, data in enumerate(train_loader, 0):
            X, y = data

            optimizer.zero_grad()

            yout = net(X)
            _, pred = torch.max(yout.data, 1)
            total_acc += (pred == y).sum().item() / y.size(0)

            loss = criterion(yout, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
        accs.append(total_acc)
        if epoch % 2 == 0:    # print every 2000 mini-batches
            print('[epoch %d] loss: %.3f acc: %.3f' %
                  (epoch + 1, total_loss, total_acc))

    ep = np.arange(start=0, stop=epochs, step=1)
    plt.subplot(1, 2, 1)
    plt.plot(ep, losses, 'ko-')
    plt.title('Loss vs Epochs')

    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.subplot(1, 2, 2)
    plt.plot(ep, accs, 'r.-')
    plt.title('Accuracy vs Epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')

    plt.savefig("../out/q6/cnn_loss-{:.3f}_acc-{:.3f}_epocs-{}_lr-{}_batch-{}.png"
                .format(total_loss, total_acc, epochs, lr_rate, batch_size))

    plt.show()

    total = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            X, y = data
            yout = net(X)
            _, pred = torch.max(yout.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    print(f"Validation accuracy: {correct / total}")


if __name__ == "__main__":
    # MLP
    # q6_1_1()

    # CNN
    q6_1_2()
