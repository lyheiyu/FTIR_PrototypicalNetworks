import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class IncrementalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.encoder(x)


class IncrementalFewShotLearning():
    def __init__(self, input_size, hidden_size, output_size, num_classes=5, num_shots=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.classes = []
        self.model = None

    def add_classes(self, data):
        num_new_classes = len(data) // self.num_shots
        self.classes += [i + len(self.classes) for i in range(num_new_classes)]
        dataset = IncrementalDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.num_shots, shuffle=True)
        if self.model is None:
            self.model = PrototypicalNetwork(self.input_size, self.hidden_size, self.output_size)
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optim.Adam(self.model.parameters())
        for epoch in range(10):
            self.model.train()
            for batch in dataloader:
                inputs = batch[:, :-1]
                labels = batch[:, -1].long() + len(self.classes) - num_new_classes
                prototypes = torch.zeros(self.num_classes, self.hidden_size)
                num_samples = torch.zeros(self.num_classes)
                for i in range(len(inputs)):
                    class_idx = labels[i]
                    prototypes[class_idx] += self.model(inputs[i])
                    num_samples[class_idx] += 1
                for i in range(self.num_classes):
                    prototypes[i] /= num_samples[i]
                prototypes = prototypes.to('cuda')
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = self.model(inputs)
                distances = torch.cdist(outputs, prototypes)
                loss = nn.CrossEntropyLoss()(distances, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, inputs):
        self.model.eval()
        prototypes = torch.zeros(self.num_classes, self.hidden_size)
        num_samples = torch.zeros(self.num_classes)
        for i in range(len(self.classes)):
            prototypes[i] = self.model(inputs[i])
            num_samples[i] = 1
        for i in range(self.num_classes):
            prototypes[i] /= num_samples[i]
        prototypes = prototypes.to('cuda')
        inputs = inputs.to('cuda')
        distances = torch.cdist(inputs, prototypes)
        _, preds = torch.min(distances, dim=1)
        return [self.classes[pred] for pred in preds]
