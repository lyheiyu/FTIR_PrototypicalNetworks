import torch
import torch.nn as nn
import torch.optim as optim
from FTIR_GenerateData import GenerateDate
import numpy as np
from sklearn.model_selection import train_test_split
# 自适应度量学习的嵌入器
def convert_to_pytorch_tensors2(data_dict):
    tensor_dict = {}
    for name, data in data_dict.items():
        if len(data.shape) == 2:  # 处理输入数据 x_train，形状 (batch_size, sequence_length)
            tensor_dict[name] = torch.tensor(data).float()  # 保持二维输入 (batch_size, sequence_length)
        elif len(data.shape) == 1:  # 处理标签数据 y_train，形状 (batch_size,)
            tensor_dict[name] = torch.tensor(data).long()  # 转换为 long 类型
        else:
            raise ValueError(f"Data {name} has unexpected shape: {data.shape}")
    return tensor_dict

class AdaptiveMetricNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdaptiveMetricNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

    # 计算马氏距离，加入协方差矩阵正则化
    def mahalanobis_distance(self, x, y, cov_matrix, epsilon=1e-3):
        diff = x - y
        # 增加正则化以防止协方差矩阵奇异
        regularized_cov_matrix = cov_matrix + torch.eye(cov_matrix.size(0)) * epsilon
        inv_cov_matrix = torch.inverse(regularized_cov_matrix)

        # 计算马氏距离
        distance = torch.sqrt(torch.clamp(torch.sum(diff @ inv_cov_matrix * diff, dim=-1), min=0))
        return distance


# 计算每个类的原型和协方差矩阵
def compute_prototypes_and_covariances(encoder, support_set, support_labels, num_classes):
    embeddings = encoder(support_set)
    print(
        f"Embeddings stats: mean={embeddings.mean()}, std={embeddings.std()}, min={embeddings.min()}, max={embeddings.max()}")
    prototypes = []
    cov_matrices = []

    for i in range(num_classes):
        class_embeddings = embeddings[support_labels == i]
        if class_embeddings.shape[0] == 0:
            print(f"No samples found for class {i}")
            continue

        class_prototype = class_embeddings.mean(dim=0)
        prototypes.append(class_prototype)

        centered_embeddings = class_embeddings - class_prototype
        cov_matrix = centered_embeddings.t() @ centered_embeddings / class_embeddings.shape[0]
        cov_matrices.append(cov_matrix)

    return torch.stack(prototypes), cov_matrices

# 查询集的分类逻辑（返回 logits，而不是类标签）
def classify_query(encoder, query_set, prototypes, cov_matrices):
    query_embeddings = encoder(query_set)
    logits = []

    for query_embedding in query_embeddings:
        distances = []
        for class_index, prototype in enumerate(prototypes):
            # 使用马氏距离
            cov_matrix = cov_matrices[class_index]
            dist = encoder.mahalanobis_distance(query_embedding, prototype, cov_matrix)
            distances.append(dist)

        # 将所有类的距离存入 logits
        logits.append(torch.stack(distances))

    # 将所有查询样本的距离拼接为 [batch_size, num_classes]
    logits = torch.stack(logits)

    return logits

# 训练步骤
def train_step(encoder, support_set, support_labels, query_set, query_labels, num_classes, optimizer, criterion):
    encoder.train()
    optimizer.zero_grad()

    # 计算类原型和协方差矩阵
    prototypes, cov_matrices = compute_prototypes_and_covariances(encoder, support_set, support_labels, num_classes)

    # 分类查询集
    logits = classify_query(encoder, query_set, prototypes, cov_matrices)
    logits = logits.view(query_labels.shape[0], num_classes)  # 调整 logits 形状以匹配标签形状

    # 计算损失
    loss = criterion(logits, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

# 推理阶段
def infer(encoder, support_set, support_labels, query_set, query_labels, num_classes):
    encoder.eval()
    with torch.no_grad():
        prototypes, cov_matrices = compute_prototypes_and_covariances(encoder, support_set, support_labels, num_classes)
        logits = classify_query(encoder, query_set, prototypes, cov_matrices)
        predicted_labels = torch.argmax(logits, dim=1)

    # 打印预测结果
    for i, predicted_label in enumerate(predicted_labels):
        print(f"Query {i + 1}: Predicted label = {predicted_label.item()}, True label = {query_labels[i].item()}")

# 数据生成部分
GenData = GenerateDate()
firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = GenData.getData()

x_train1, y_train1, x_test1, y_test1 = GenData.dataAugmenation(firstData, pid1, wavenumber, pname1, 1)
x_train2, y_train2, x_test2, y_test2 = GenData.dataAugmenation2(secondData, pid2, wavenumber, pname2, 1)

wavenumber4, forthData, pid4, pname4 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')
x_train4, x_test4, y_train4, y_test4 = train_test_split(forthData, pid4, test_size=0.7, random_state=1)

wavenumber5, fifthData, pid5, pname5 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
x_train5, x_test5, y_train5, y_test5 = train_test_split(fifthData, pid5, test_size=0.7, random_state=1)

# 将数据转换为 PyTorch 张量
data_dict = {
    'x_train1': x_train1, 'y_train1': y_train1,
    'x_train2': x_train2, 'y_train2': y_train2,
    'x_test1': x_test1, 'y_test1': y_test1,
    'x_test2': x_test2, 'y_test2': y_test2,
    'x_train4': x_train4, 'y_train4': y_train4,
    'x_train5': x_train5, 'y_train5': y_train5,
    'x_test4': x_test4, 'y_test4': y_test4,
    'x_test5': x_test5, 'y_test5': y_test5,
}

tensor_dict = convert_to_pytorch_tensors2(data_dict)

# 访问转换后的张量
x_train1, y_train1 = tensor_dict['x_train1'], tensor_dict['y_train1']
x_test1, y_test1 = tensor_dict['x_test1'], tensor_dict['y_test1']
x_train2, y_train2 = tensor_dict['x_train2'], tensor_dict['y_train2']
x_test2, y_test2 = tensor_dict['x_test2'], tensor_dict['y_test2']

# 模型、优化器、损失函数定义
input_dim = x_train1.shape[1]
embedding_dim = 128
num_classes = len(np.unique(pid1))  # 根据 pid1 的类别数量
encoder = AdaptiveMetricNetwork(input_dim, embedding_dim)
optimizer = optim.Adam(encoder.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
# def sample_from_set(support_set, support_labels, query_set, query_labels, batch_size):
#     """
#     从 support set 和 query set 中随机采样一批数据用于训练。
#     """
#     # 随机从 support set 中采样
#     support_indices = torch.randperm(len(support_set))[:batch_size]
#     support_sampled = support_set[support_indices]
#     support_labels_sampled = support_labels[support_indices]
#
#     # 随机从 query set 中采样
#     query_indices = torch.randperm(len(query_set))[:batch_size]
#     query_sampled = query_set[query_indices]
#     query_labels_sampled = query_labels[query_indices]
#
#     return support_sampled, support_labels_sampled, query_sampled, query_labels_sampled
#
# # 训练步骤
# def train_step_with_sampling(encoder, support_set, support_labels, query_set, query_labels,
#                              num_classes, optimizer, criterion, batch_size):
#     encoder.train()
#     optimizer.zero_grad()
#
#     # 随机采样一批数据
#     support_sampled, support_labels_sampled, query_sampled, query_labels_sampled = sample_from_set(
#         support_set, support_labels, query_set, query_labels, batch_size)
#
#     # 计算类原型和协方差矩阵
#     prototypes, cov_matrices = compute_prototypes_and_covariances(encoder, support_sampled, support_labels_sampled,
#                                                                   num_classes)
#
#     # 分类查询集
#     logits = classify_query(encoder, query_sampled, prototypes, cov_matrices)
#
#     # 确保 logits 的形状为 [batch_size, num_classes]
#     print(f"logits shape: {logits.shape}")  # 检查 logits 形状
#     assert logits.shape == (batch_size, num_classes), "Logits shape mismatch"
#
#     # 计算损失
#     loss = criterion(logits, query_labels_sampled)
#
#     # 反向传播并裁剪梯度
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
#
#     optimizer.step()
#
#     return loss.item()
def stratified_sample_from_set(support_set, support_labels, query_set, query_labels, batch_size, num_classes):
    """
    分层采样函数，从 support_set 和 query_set 中按类别分别采样，确保每个类别都有样本。
    """
    # 创建采样后的集合
    support_sampled = []
    support_labels_sampled = []
    query_sampled = []
    query_labels_sampled = []

    # 计算每个类的样本数量
    support_per_class = batch_size // num_classes
    query_per_class = batch_size // num_classes

    for class_index in range(num_classes):
        # 获取该类别的索引
        support_class_indices = torch.where(support_labels == class_index)[0]
        query_class_indices = torch.where(query_labels == class_index)[0]

        # 如果该类别的样本不足以满足需要，使用所有样本
        if len(support_class_indices) < support_per_class:
            print(f"Class {class_index} has insufficient support samples. Using all available samples.")
            sampled_support_indices = support_class_indices
        else:
            sampled_support_indices = support_class_indices[torch.randperm(len(support_class_indices))[:support_per_class]]

        if len(query_class_indices) < query_per_class:
            print(f"Class {class_index} has insufficient query samples. Using all available samples.")
            sampled_query_indices = query_class_indices
        else:
            sampled_query_indices = query_class_indices[torch.randperm(len(query_class_indices))[:query_per_class]]

        # 采样支持集和查询集
        support_sampled.append(support_set[sampled_support_indices])
        support_labels_sampled.append(support_labels[sampled_support_indices])
        query_sampled.append(query_set[sampled_query_indices])
        query_labels_sampled.append(query_labels[sampled_query_indices])

    # 将采样结果拼接成张量
    support_sampled = torch.cat(support_sampled, dim=0)
    support_labels_sampled = torch.cat(support_labels_sampled, dim=0)
    query_sampled = torch.cat(query_sampled, dim=0)
    query_labels_sampled = torch.cat(query_labels_sampled, dim=0)

    return support_sampled, support_labels_sampled, query_sampled, query_labels_sampled


def train_step_with_stratified_sampling(encoder, support_set, support_labels, query_set, query_labels,
                                        num_classes, optimizer, criterion, batch_size):
    encoder.train()
    optimizer.zero_grad()

    # 使用分层采样
    support_sampled, support_labels_sampled, query_sampled, query_labels_sampled = stratified_sample_from_set(
        support_set, support_labels, query_set, query_labels, batch_size, num_classes)

    # 计算类原型和协方差矩阵
    prototypes, cov_matrices = compute_prototypes_and_covariances(encoder, support_sampled, support_labels_sampled,
                                                                  num_classes)

    # 分类查询集
    logits = classify_query(encoder, query_sampled, prototypes, cov_matrices)

    # 确保 logits 的形状为 [batch_size, num_classes]
    print(f"logits shape: {logits.shape}")  # 确认 logits 形状
    assert logits.shape == (batch_size, num_classes), "Logits shape mismatch"

    # 计算损失
    loss = criterion(logits, query_labels_sampled)

    # 反向传播并裁剪梯度
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()
# 设置批次大小
batch_size = 16  # 控制采样大小，您可以根据内存情况调整

# 训练过程
for epoch in range(1000):
    loss = train_step_with_stratified_sampling(encoder, x_train1, y_train1, x_test1, y_test1, num_classes, optimizer, criterion, batch_size)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# 推理
infer(encoder, x_train1, y_train1, x_test1, y_test1, num_classes)
