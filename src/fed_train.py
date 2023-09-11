import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import client
import LeNet_5
from poisoned_MNIST import PoisonedMNIST
from rofl import norm_bound

# Hyperparameters
args = {
    'batch_size': 32,
    'test_batch_size': 1000,
    'rounds': 5,
    'epochs_per_round': 1,
    'lr': 0.01,
    'momentum': 0.5,
    'log_interval': 10,
}

num_clients = 10
poisoned_clients = 5
federated_dataset_size = 0
use_rofl = True

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Load MNIST dataset
datadir = '../data/'

train_dataset = torchvision.datasets.MNIST(datadir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))

test_dataset = torchvision.datasets.MNIST(datadir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))

poisoned_dataset = PoisonedMNIST(datadir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))

# split training dataset into num_clients parts
lengths = [int(len(train_dataset)/num_clients) for i in range(num_clients)]
print(lengths)
federated_train_dataset = torch.utils.data.random_split(train_dataset, lengths)

# split poisoned dataset into num_clients parts
lengths = [int(len(poisoned_dataset)/num_clients) for i in range(num_clients)]
federated_poisoned_dataset = torch.utils.data.random_split(poisoned_dataset, lengths)

train_loaders = []
poisoned_loaders = []

for i in range(num_clients - poisoned_clients):
    train_loaders.append(torch.utils.data.DataLoader(
        federated_train_dataset[i], batch_size=args['batch_size'], shuffle=True))

for i in range(poisoned_clients):
    poisoned_loaders.append(torch.utils.data.DataLoader(
        federated_poisoned_dataset[i], batch_size=args['batch_size'], shuffle=True))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args['test_batch_size'], shuffle=True)

# Initialize clients
clients = []
for i in range(1, num_clients + 1):
    if i <= poisoned_clients:
        clients.append(client.Client(i, args, poisoned_loaders[i - 1], test_loader))
    else:
        clients.append(client.Client(i, args, train_loaders[i - poisoned_clients - 1], test_loader))

# Test
def test(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += torch.nn.functional.cross_entropy(output, target.to(device), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))
    

# Run
acc = []
list_norm = []
for i in range(len(clients)):
    list_norm.append([])

for round in range(1, args['rounds'] + 1):
    model_updates = []
    for client in clients:
        model_updates.append(client.train(round))
        client.test()

    # Average
    avg_model = {}
    if use_rofl:
        num_accepted_clients = 0
        weights = []
        for i in range(len(model_updates)):
            n = norm_bound(model_updates[i])
            list_norm[i].append(n)
            print("Client {}: {}".format(i + 1, n))
            if n < 100:
                weights.append(model_updates[i])
                num_accepted_clients += 1
                print("Client {} Norm bound satisfied".format(i + 1))
            else:
                print("Client {} Norm bound not satisfied".format(i + 1))
        print("Number of accepted clients: {}".format(num_accepted_clients))
        if num_accepted_clients == 0:
            print("No accepted clients, skipping round")
            avg_model = clients[num_clients - 1].model.state_dict()
        else:
            for name in weights[0].keys():
                avg_model[name] = sum([param[name].data for param in weights]) / len(weights)
    else:
        for name in model_updates[0].keys():
            avg_model[name] = sum([param[name].data for param in model_updates]) / len(model_updates)

    # Update
    for client in clients:
        client.model.load_state_dict(avg_model)

    global_model = LeNet_5.LeNet_5().to(device)
    global_model.load_state_dict(avg_model)
    res = test(global_model)
    print("Round: {} Test accuracy = {}\n".format(round, res))
    acc.append(res)

print("Total accuracy: ", acc)
for client in clients:
    print("Client {} accuracy: {}".format(client.id, client.get_acc()))

if use_rofl:
    print("L2 Norm")
    for i in range(len(clients)):
        print("Client {}: {}".format(i + 1, list_norm[i]))

with open("../result.txt", "a") as f:
    f.write("\n#c{} p{} r{} e{} lr{} m{}\n".format(num_clients, poisoned_clients, args['rounds'], args['epochs_per_round'], args['lr'], args['momentum']))
    f.write("Total accuracy: {}\n".format(acc))
    for client in clients:
        f.write("Client {} accuracy: {}\n".format(client.id, client.get_acc()))
    if use_rofl:
        f.write("L2 Norm\n")
        for i in range(len(clients)):
            f.write("Client {}: {}\n".format(i + 1, list_norm[i]))
        f.write("\n")
    f.close()