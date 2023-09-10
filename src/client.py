import torch
import torch.optim as optim
import numpy as np
import LeNet_5

class Client:
    def __init__(self, id, args, train_loader, test_loader):
        self.id = id
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet_5.LeNet_5().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])
        self.acc = []
        self.loss = []

    def train(self, round):
        for epoch in range(1, self.args['epochs_per_round'] + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data.to(self.device))
                loss = torch.nn.functional.cross_entropy(output, target.to(self.device))
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.args['log_interval'] == 0:
                    print('Client: {} Train Round: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            self.id, round, epoch, batch_idx * len(data), len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader), loss.item()))
                    self.loss.append(loss.item())
        return self.model.state_dict()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data.to(self.device))
                test_loss += torch.nn.functional.cross_entropy(output, target.to(self.device), reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.to(self.device).view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        print('\nClient: {} Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.id, test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        self.acc.append(100. * correct / len(self.test_loader.dataset))

    def get_acc(self):
        return self.acc