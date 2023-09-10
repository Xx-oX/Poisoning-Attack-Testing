import syft as sy
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import LeNet_5

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

clients = []
num_clients = 10

for i in range(num_clients):
    clients.append(sy.VirtualWorker(id="client{}".format(i)))

args = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 5,
    'lr': 0.01,
    'momentum': 0.5,
}

# Load MNIST dataset
federated_train_loader = sy.FederatedDataLoader(
    torchvision.datasets.MNIST('../data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))
    .federate(clients), # distribute the dataset across all the workers
    batch_size=args['batch_size'], shuffle=True)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
    batch_size=args['test_batch_size'], shuffle=True)

# Train
def train(args, model, federated_train_loader, optimizer, epoch, train_loss_batch, train_loss_epoch, ids):
    model.train()
    client = None
    t_loss = 0
    total = 0

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        if client == None:
            client =  data.location.id
        elif client != data.location.id:
            client = data.location.id
        print("Processing: ", client)
        model.send(data.location) # send the model to the client
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        model.get() # get the model back from the client
        if batch_idx % 30 == 0:
            loss = loss.get() # get the loss back
        t_loss += loss
        total += 1
        train_loss_batch.append(loss.item())
        ids[client] = ids[client].append(loss.item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(federated_train_loader),
                100. * batch_idx / len(federated_train_loader), loss.item()))
        t_loss /= total
        train_loss_epoch.append(t_loss.item())
        
# Test
def test(args, model, test_loader, test_loss):
    model.eval()
    test_loss = 0
    correct = 0

    with troch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_loss.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(  
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

# Run
model = LeNet_5.LeNet_5().to(device)
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

train_loss_batch = []
train_loss_epoch = []

ids = [[]] * num_clients

test_lost = []

for epoch in range(1, args['epochs'] + 1):
    train(args, model, federated_train_loader, optimizer, epoch, train_loss_batch, train_loss_epoch, ids)
    test(args, model, test_loader, test_loss)

# Save
torch.save(model.state_dict(), '../data/fed_model.pth')

# Plot
plt.figure()
plt.plot(train_loss_batch)
plt.title('Training Loss vs. Batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('../data/training_loss_batch.png')

plt.figure()
plt.plot(train_loss_epoch)
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../data/training_loss_epoch.png')

plt.figure()
for i in range(num_clients):
    plt.plot(ids[i])
plt.legend()
plt.title('Client Loss vs. Batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('../data/client_loss.png')

