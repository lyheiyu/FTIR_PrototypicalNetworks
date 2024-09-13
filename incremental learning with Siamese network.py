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


class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class IncrementalFewShotLearning():
    def __init__(self, input_size, hidden_size, num_classes=5, num_shots=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.classes = []
        self.model = None

    def add_classes(self, data):
        # Calculate the number of new classes to add based on the number of examples in the data and the number of shots per class
        num_new_classes = len(data) // self.num_shots

        # Add the new class indices to the list of classes
        self.classes += [i + len(self.classes) for i in range(num_new_classes)]

        # Create a new dataset and dataloader for the new classes
        dataset = IncrementalDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.num_shots, shuffle=True)

        # If this is the first time adding classes, create a new model and optimizer
        if self.model is None:
            self.model = SiameseNetwork(self.input_size, self.hidden_size)
            self.optimizer = optim.Adam(self.model.parameters())
        # Otherwise, re-use the existing model and optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters())

        # Train the model for 10 epochs
        for epoch in range(10):
            self.model.train()
            for batch in dataloader:
                inputs = batch[:, :-1]
                labels = batch[:, -1].long() + len(self.classes) - num_new_classes

                # Split the inputs and labels into two sets for the siamese network
                inputs1 = inputs[:self.num_shots // 2]
                inputs2 = inputs[self.num_shots // 2:]
                labels1 = labels[:self.num_shots // 2]
                labels2 = labels[self.num_shots // 2:]

                # Forward pass through the siamese network and calculate the pairwise distance between the outputs
                outputs1, outputs2 = self.model(inputs1, inputs2)
                distances = nn.PairwiseDistance()(outputs1, outputs2)

                # Compute the loss as the cross-entropy loss between the pairwise distances and whether the two inputs belong to the same class
                loss = nn.CrossEntropyLoss()(distances, (labels1 == labels2).long())

                # Zero out the gradients, perform backpropagation, and update the model parameters
                self.optimizer.zero_grad()

