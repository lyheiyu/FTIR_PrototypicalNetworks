import torch
import torch.nn as nn
import torch.optim as optim
from FTIR_GenerateData import GenerateDate
from sklearn.model_selection import train_test_split
import numpy as np
# 自适应度量学习的嵌入器
def convert_to_pytorch_tensors(data_dict):
    tensor_dict = {}
    for name, data in data_dict.items():
        if len(data.shape) == 2:  # 处理输入数据 x_train，形状 (batch_size, sequence_length)
            tensor_dict[name] = torch.tensor(data).float().unsqueeze(1)  # 增加通道维度，变为 (batch_size, 1, sequence_length)
        elif len(data.shape) == 1:  # 处理标签数据 y_train，形状 (batch_size,)
            tensor_dict[name] = torch.tensor(data).long()  # 转换为 long 类型
        else:
            raise ValueError(f"Data {name} has unexpected shape: {data.shape}")
    return tensor_dict
class AdaptiveMetricNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(AdaptiveMetricNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),  # 输入通道为1，输出通道为64
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, embedding_dim)  # 输出嵌入维度
        )

    def forward(self, x):
        return self.encoder(x)

    # 计算马氏距离，加入协方差矩阵正则化
    def mahalanobis_distance(self, x, y, cov_matrix, epsilon=1e-6):
        diff = x - y
        # 给协方差矩阵加上一个小的对角矩阵，确保它是可逆的
        regularized_cov_matrix = cov_matrix + torch.eye(cov_matrix.size(0)) * epsilon
        inv_cov_matrix = torch.inverse(regularized_cov_matrix)  # 计算协方差矩阵的逆
        return torch.sqrt(torch.clamp(torch.sum(diff @ inv_cov_matrix * diff, dim=-1), min=0))


# 计算每个类的原型和协方差矩阵
def compute_prototypes_and_covariances(encoder, support_set, support_labels, num_classes):
    embeddings = encoder(support_set)  # 获取支持集嵌入
    prototypes = []
    cov_matrices = []

    for i in range(num_classes):
        # 获取该类的所有样本嵌入
        class_embeddings = embeddings[support_labels == i]

        # 计算类原型（均值）
        class_prototype = class_embeddings.mean(dim=0)
        prototypes.append(class_prototype)

        # 计算协方差矩阵
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
            cov_matrix = cov_matrices[class_index]
            dist = encoder.mahalanobis_distance(query_embedding, prototype, cov_matrix)
            distances.append(dist)
        logits.append(torch.stack(distances))  # 返回每个类的距离作为 logits
    # print(f"logits shape: {logits.shape}")
    return torch.stack(logits)  # 形状应为 (batch_size, num_classes)


# 训练步骤
def train_step(encoder, support_set, support_labels, query_set, query_labels, num_classes, optimizer, criterion):
    encoder.train()
    optimizer.zero_grad()

    # 计算类原型和协方差矩阵
    prototypes, cov_matrices = compute_prototypes_and_covariances(encoder, support_set, support_labels, num_classes)

    # 分类查询集，返回的是 logits (未归一化的距离)
    logits = classify_query(encoder, query_set, prototypes, cov_matrices)
    print(f"logits shape: {logits.shape}")  # 期望形状是 (batch_size, num_classes)
    print(f"query_labels shape: {query_labels.shape}")  # 期望形状是 (batch_size,)
    # 计算损失
    # query_labels = query_labels.view(-1)

    # 计算损失
    # loss = criterion(logits, query_labels)
    loss = criterion(logits, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


# 生成随机支持集和查询集数据
def prepare_data(num_classes, support_size, query_size, sequence_length):
    support_set = torch.randn(support_size, sequence_length, 1)  # 支持集
    support_labels = torch.randint(0, num_classes, (support_size,))  # 支持集标签
    query_set = torch.randn(query_size, sequence_length, 1)  # 查询集
    query_labels = torch.randint(0, num_classes, (query_size,))  # 查询集标签

    # 将数据从 (batch_size, sequence_length, channels) 转为 (batch_size, channels, sequence_length)
    support_set = support_set.permute(0, 2, 1)
    query_set = query_set.permute(0, 2, 1)

    return support_set, support_labels, query_set, query_labels

def prepare_data2(num_classes, support_size, query_size, sequence_length):
    support_set = torch.randn(support_size, sequence_length, 1)  # 支持集
    support_labels = torch.randint(0, num_classes, (support_size,))  # 支持集标签
    query_set = torch.randn(query_size, sequence_length, 1)  # 查询集
    query_labels = torch.randint(0, num_classes, (query_size,))  # 查询集标签

    # 将数据从 (batch_size, sequence_length, channels) 转为 (batch_size, channels, sequence_length)
    support_set = support_set.permute(0, 2, 1)
    query_set = query_set.permute(0, 2, 1)

    return support_set, support_labels, query_set, query_labels
# 模型和优化器设置
embedding_dim = 128  # 嵌入维度
num_classes = 5  # 类别数量

encoder = AdaptiveMetricNetwork(embedding_dim)
optimizer = optim.Adam(encoder.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 准备数据
# support_set, support_labels, query_set, query_labels = prepare_data(num_classes, support_size=20, query_size=10,
#                                                                     sequence_length=200)
GenData=GenerateDate()
# firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()
firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = GenData.getData()

print(firstData)
print(secondData)
x_train1,y_train1,x_test1,y_test1=GenData.dataAugmenation(firstData,pid1,wavenumber,pname1,1)

#x_train3, y_train3, x_test3, y_test3 = GenData.dataAugmenation2(thirdData, pid3, wavenumber, pname3, 1)
# # # fileName='FTIR_PLastics500_c4.csv'
wavenumber4, forthData, pid4, pname4 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')
wavenumber5, fifthData, pid5, pname5 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
# #x_train3, x_test3, y_train3, y_test3 = train_test_split(thirdData, pid3, test_size=0.3, random_state=1)
# # #x_train1, x_test1, y_train1, y_test1 = train_test_split(firstData, pid1, test_size=0.3, random_state=1)
x_train2, y_train2, x_test2, y_test2 = GenData.dataAugmenation2(secondData, pid2, wavenumber, pname2, 1)
# # x_train4, y_train4, x_test4, y_test4 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
#x_train2, x_test2, y_train2, y_test2 = train_test_split(secondData, pid2, test_size=0.3, random_state=1)
# print(forthData.shape)
# x_train1, y_train1, x_test1, y_test1 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
x_train4, x_test4, y_train4, y_test4 = train_test_split(forthData, pid4, test_size=0.7, random_state=1)
x_train5, x_test5, y_train5, y_test5 = train_test_split(fifthData, pid5, test_size=0.7, random_state=1)
for item in x_train1:
    if np.any(np.isnan(item)) or np.any(np.isinf(item)):
        print('x_train1',item)

for item in x_train2:
    if np.any(np.isnan(item)) or np.any(np.isinf(item)):
        print('x_train2',item)
input_shape = 2000
num_tasks = 3
num_classes_per_task = [11, 4]
# x_train1 = torch.tensor(x_train1).float().unsqueeze(1)  # 转换为浮点型张量，并在中间加一个通道维度
#
# # 处理 y_train1，将其转换为 PyTorch 长整型张量
# y_train1 = torch.tensor(y_train1).long()
# x_train1 = torch.tensor(x_test1).float().unsqueeze(1)  # 转换为浮点型张量，并在中间加一个通道维度
#
# # 处理 y_train1，将其转换为 PyTorch 长整型张量
# y_train1 = torch.tensor(y_test1).long()
support_data_list = [x_train1, x_train2]

support_labels_list = [y_train1, y_train2]
query_data_list = [x_test1, x_test2]
#
query_labels_list = [y_test1, y_test2]

data_dict = {
    'x_train1': x_train1,
    'y_train1': y_train1,
    'x_train2': x_train2,
    'y_train2': y_train2,
    'x_test1': x_test1,
    'y_test1': y_test1,
    'x_test2': x_test2,
    'y_test2': y_test2,
    'x_train4': x_train4,
    'y_train4': y_train4,
    'x_train5': x_train5,
    'y_train5': y_train5,
    'x_test4': x_test4,
    'y_test4': y_test4,
    'x_test5': x_test5,
    'y_test5': y_test5,


}

# 调用函数进行转换
tensor_dict = convert_to_pytorch_tensors(data_dict)

# 访问转换后的张量
x_train1 = tensor_dict['x_train1']
y_train1 = tensor_dict['y_train1']
x_train2 = tensor_dict['x_train2']
y_train2 = tensor_dict['y_train2']
x_test1 = tensor_dict['x_test1']
y_test1 = tensor_dict['y_test1']
x_test2 = tensor_dict['x_test2']
y_test2 = tensor_dict['y_test2']
support_set=x_train1
support_labels=y_train1
query_set=x_test1
query_labels=y_test1
num_classes=len(np.unique(pid1))
# 训练模型

# query_labels = query_labels.view(-1)
for epoch in range(100):
    loss = train_step(encoder, support_set, support_labels, query_set, query_labels, num_classes, optimizer, criterion)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")


# 推理阶段
def infer(encoder, new_support_set, new_support_labels, new_query_set, new_query_labels, num_classes):
    encoder.eval()
    with torch.no_grad():
        # 计算新支持集的类原型和协方差矩阵
        new_prototypes, new_cov_matrices = compute_prototypes_and_covariances(encoder, new_support_set,
                                                                              new_support_labels, num_classes)

        # 分类新查询集
        new_logits = classify_query(encoder, new_query_set, new_prototypes, new_cov_matrices)
        new_predicted_labels = torch.argmax(new_logits, dim=1)  # 选择距离最小的类作为预测结果

    # 打印预测结果
    for i, predicted_label in enumerate(new_predicted_labels):
        print(f"Query {i + 1}: Predicted label = {predicted_label.item()}, True label = {new_query_labels[i].item()}")


# 使用新的支持集和查询集进行推理
# new_support_set, new_support_labels, new_query_set, new_query_labels = prepare_data(num_classes, support_size=20,
#                                                                                     query_size=10, sequence_length=200)
x_train4= tensor_dict['x_train4']
y_train4 = tensor_dict['y_train4']
x_train5 = tensor_dict['x_train5']
y_train5 = tensor_dict['y_train5']
x_test4= tensor_dict['x_test4']
y_test4 = tensor_dict['y_test4']
x_test5 = tensor_dict['x_test5']
y_test5 = tensor_dict['y_test5']

new_support_set=x_train4
new_support_labels=y_train4
new_query_set=x_test4
new_query_labels=y_test4
new_num_classes=len(np.unique(pid4))

infer(encoder, new_support_set, new_support_labels, new_query_set, new_query_labels, new_num_classes)
