import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import LeNet_5

# Hyperparameters
num_epochs = 5
batch_size = 64
batch_size_test = 1000
lr = 0.01
momentum = 0.5
log_interval = 10

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Load MNIST dataset
datadir = '../data/'
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(datadir, train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(datadir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size_test, shuffle=True)

# Initialize model
model = LeNet_5.LeNet_5().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(num_epochs + 1)]
acc = []

# Train
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = torch.nn.functional.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), '../data/model.pth')
            torch.save(optimizer.state_dict(), '../data/optimizer.pth')

# Test
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += torch.nn.functional.cross_entropy(output, target.to(device), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    acc.append(100. * correct.cpu().numpy() / len(test_loader.dataset))

# Run
test()
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

print(acc)

# Plot
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.savefig('../data/loss.png')

fig = plt.figure()
plt.plot(acc, color='blue')
plt.legend(['Test Accuracy'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
fig.savefig('../data/accuracy.png')
