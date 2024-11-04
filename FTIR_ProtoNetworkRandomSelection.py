import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler

class ProtoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, support, query, num_classes, num_support_per_class):
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)

        # 计算原型
        prototypes = []
        for i in range(num_classes):
            class_support_embeddings = support_embeddings[i * num_support_per_class:(i + 1) * num_support_per_class]
            prototypes.append(class_support_embeddings.mean(dim=0))
        prototypes = torch.stack(prototypes)

        # 计算距离
        distances = torch.cdist(query_embeddings, prototypes)

        # 输出 logits
        logits = -distances
        return logits

def loadFTIRData():
    # 加载您的数据
    features = your_ftir_data  # 替换为您的实际数据
    labels = your_labels       # 替换为您的实际标签

    # 数据标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 转换为张量
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    labels -= labels.min()

    return features, labels

def randomGenerateForProtoNet(features, labels, num_classes, num_support_per_class, num_query_per_class):
    support_inputs = []
    query_inputs = []
    query_labels = []

    for class_id in range(num_classes):
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
        class_samples = features[class_indices]

        if class_samples.size(0) < num_support_per_class + num_query_per_class:
            raise ValueError(f"类别 {class_id} 的样本不足")

        indices = torch.randperm(class_samples.size(0))
        support_indices = indices[:num_support_per_class]
        query_indices = indices[num_support_per_class:num_support_per_class + num_query_per_class]

        support_inputs.append(class_samples[support_indices])
        query_inputs.append(class_samples[query_indices])
        query_labels.extend([class_id] * num_query_per_class)

    support_input = torch.cat(support_inputs, dim=0)
    query_input = torch.cat(query_inputs, dim=0)
    query_labels = torch.LongTensor(query_labels)

    return support_input, query_input, query_labels

def trainProtoNet(model, optimizer, support_input, query_input, query_labels, num_classes, num_support_per_class):
    model.train()
    optimizer.zero_grad()

    logits = model(support_input, query_input, num_classes, num_support_per_class)
    loss = F.cross_entropy(logits, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == '__main__':
    hidden_dim = 128
    num_classes = ...  # 设置为您的类别数量
    num_support_per_class = 5
    num_query_per_class = 15
    lr = 0.001
    epochs = 100

    features, labels = loadFTIRData()
    model = ProtoNet(features.shape[1], hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss_sum = 0
        num_iterations = 10
        for _ in range(num_iterations):
            support_input, query_input, query_labels = randomGenerateForProtoNet(
                features, labels, num_classes, num_support_per_class, num_query_per_class)
            loss = trainProtoNet(model, optimizer, support_input, query_input, query_labels,
                                 num_classes, num_support_per_class)
            loss_sum += loss
        avg_loss = loss_sum / num_iterations
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
