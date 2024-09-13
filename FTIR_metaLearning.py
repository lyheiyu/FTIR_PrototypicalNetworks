import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from copy import deepcopy
from utils import utils
print(torch.cuda.get_device_name())

trainset = np.load("train_data.npy")
testset = np.load("test_data.npy")

print(trainset.shape, testset.shape)

meta_batch_size = 32
alpha = 0.04
beta = 0.0001


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64 * 4 * 4, 128)
        self.out = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)

        x = F.relu(self.bn4(self.conv4(x)))

        x = x.reshape(-1, 64 * 4 * 4)
        x = self.fc(x)
        x = self.out(x)

        return x


def task_sample(mode):
    set_len = 1200 if mode == "train" else 423
    curset = trainset if mode == "train" else testset
    categories = random.sample(range(set_len), 5)
    # categories = [0, 1, 3, 50, 100]
    spt_x = None
    qry_x = None
    spt_y = torch.tensor([0, 1, 2, 3, 4])
    qry_y = torch.tensor([0, 1, 2, 3, 4])
    for _ in range(5):
        i = categories[_]
        j, k = random.sample(range(20), 2)

        cur_spt = torch.from_numpy(curset[i][j])
        cur_qry = torch.from_numpy(curset[i][k])
        # print("category:", i, "numbers:", j, k)
        if _ == 0:
            spt_x = cur_spt.unsqueeze(0)
            qry_x = cur_qry.unsqueeze(0)
        else:
            spt_x = torch.cat([spt_x, cur_spt.unsqueeze(0)], dim=0)
            qry_x = torch.cat([qry_x, cur_qry.unsqueeze(0)], dim=0)
    # print(spt_x.shape, spt_y.shape, qry_x.shape, qry_y.shape)
    return spt_x, spt_y, qry_x, qry_y


class BaseLearner():
    def __init__(self, learning_rate, model):
        self.model = deepcopy(model)
        self.alpha = learning_rate
        self.opt = None

    def update(self, model, learning_rate):
        self.model = deepcopy(model)
        self.opt = optim.SGD(self.model.parameters(), lr=learning_rate)

    def train_task(self):
        correct = 0
        self.model = self.model.cuda()
        spt_x, spt_y, qry_x, qry_y = task_sample("train")
        spt_x, spt_y, qry_x, qry_y = spt_x.cuda(), spt_y.cuda(), qry_x.cuda(), qry_y.cuda()
        # paras = [ele for ele in self.model.parameters()]

        ret = self.model(spt_x)
        loss = F.cross_entropy(ret, spt_y)
        self.opt.zero_grad()
        loss.backward()
        # grads = [ele.grad for ele in self.model.parameters()]
        self.opt.step()

        ret = self.model(qry_x)
        loss = F.cross_entropy(ret, qry_y)
        self.opt.zero_grad()
        loss.backward()

        correct += ret.argmax(dim=1).eq(qry_y).sum().item()

        self.model = self.model.cpu()
        # loss, grads, correct numbers
        return loss.item(), [ele.grad for ele in self.model.parameters()], correct

    def test_task(self):
        correct = 0
        self.model = self.model.cuda()
        spt_x, spt_y, qry_x, qry_y = task_sample("test")
        spt_x, spt_y, qry_x, qry_y = spt_x.cuda(), spt_y.cuda(), qry_x.cuda(), qry_y.cuda()

        for i in range(1):
            ret = self.model(spt_x)
            loss = F.cross_entropy(ret, spt_y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        ret = self.model(qry_x)
        loss = F.cross_entropy(ret, qry_y)
        # print("Loss:", loss.item())
        correct += ret.argmax(dim=1).eq(qry_y).sum().item()
        self.model = self.model.cpu()
        # print("Accuracy:", correct / 5, "\n")
        return loss.item(), correct


class MetaLearner():
    def __init__(self, learning_rate, batch_size):
        self.model = Net()
        self.beta = learning_rate
        self.meta_batch_size = batch_size
        self.BL = BaseLearner(alpha, self.model)
        self.train_losses = list()

    def train_one_step(self):
        grads = list()
        losses = list()
        total_correct = 0
        for batch_id in range(self.meta_batch_size):
            self.BL.update(self.model, self.BL.alpha)
            cur = self.BL.train_task()
            grads.append(cur[1])
            losses.append(cur[0])
            total_correct += cur[2]
        # update the meta model
        paras = [para for para in self.model.named_parameters()]
        for batch_id in range(self.meta_batch_size):
            for i in range(len(paras)):
                # if "bn" not in paras[i][0]:
                # if batch_id == 0: print(paras[i][0])
                paras[i][1].data = paras[i][1].data - self.beta * grads[batch_id][i].data

        return sum(losses) / self.meta_batch_size, total_correct / (self.meta_batch_size * 5)

    def train(self, epochs):
        for meta_epoch in range(epochs):
            cur_loss, acc = self.train_one_step()
            self.train_losses.append(cur_loss)
            if (meta_epoch + 1) % 1000 == 0:
                print("Meta Training Epoch:", meta_epoch + 1)
                print("Loss:", cur_loss)
            # print("Train Accuracy:", acc)

    def test_one_step(self):
        total_correct = 0
        mp = [para for para in self.model.parameters()]
        for batch_id in range(self.meta_batch_size):
            # print("Test task:", batch_id+1)
            self.BL.update(self.model, self.BL.alpha)
            cur = self.BL.test_task()
            total_correct += cur[1]

        return total_correct / (self.meta_batch_size * 5)

    def test(self, epochs):
        for test_round in range(epochs):
            acc = self.test_one_step()
            print("Test Round:", test_round + 1)
            # print("Loss:", cur_loss)
            print("Test Accuracy:", acc)

ML = MetaLearner(beta, meta_batch_size)

ML.train(20000)
plt.plot(ML.train_losses)

ML.test(100)
