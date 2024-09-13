import tensorflow as tf

# class MAML:
#     def __init__(self, feature_extractor, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1):
#         self.feature_extractor = feature_extractor
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.num_inner_steps = num_inner_steps
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
#
#     def inner_update(self, x, y, classifier_head):
#         with tf.GradientTape() as tape:
#             features = self.feature_extractor(x)
#             logits = classifier_head(features)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
#         grads = tape.gradient(loss, classifier_head.trainable_variables)
#         updated_weights = [w - self.inner_lr * g for w, g in zip(classifier_head.trainable_variables, grads)]
#         return updated_weights
#
#     def model_with_updated_weights(self, x, classifier_head, updated_weights):
#         temp_model = self.build_temp_model(classifier_head)
#         temp_model.set_weights(updated_weights)
#         features = self.feature_extractor(x)
#         return temp_model(features)
#
#     def build_temp_model(self, classifier_head):
#         model_copy = tf.keras.models.clone_model(classifier_head)
#         model_copy.set_weights(classifier_head.get_weights())
#         return model_copy
#
#     def outer_update(self, meta_dataset):
#         outer_grads = [tf.zeros_like(var) for var in self.feature_extractor.trainable_variables]
#         for x, y, classifier_head in meta_dataset:
#             updated_weights = self.inner_update(x, y, classifier_head)
#             with tf.GradientTape() as tape:
#                 logits = self.model_with_updated_weights(x, classifier_head, updated_weights)
#                 loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
#             grads = tape.gradient(loss, self.feature_extractor.trainable_variables)
#             outer_grads = [og + g for og, g in zip(outer_grads, grads)]
#         self.optimizer.apply_gradients(zip(outer_grads, self.feature_extractor.trainable_variables))
#
#     def train(self, meta_dataset, epochs):
#         for epoch in range(epochs):
#             self.outer_update(meta_dataset)
#             print(f"Epoch {epoch + 1} completed")
#
# # 定义共享的特征提取器
# def build_feature_extractor(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=input_shape),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu')
#     ])
#     return model
#
# # 定义任务特定的分类头
# def build_classifier_head(num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     return model
#
# # 示例数据形状和任务
# input_shape = (2000,)  # 输入形状，表示每个样本有2000个特征
# num_classes_per_task = [11, 4, 6]  # 每个任务的类别数量
#
# # 共享的特征提取器
# feature_extractor = build_feature_extractor(input_shape)
#
# # 为每个任务生成一个分类头
# classifier_heads = [build_classifier_head(num_classes) for num_classes in num_classes_per_task]
#
# # 假设你有多个任务，每个任务的数据集 (x_train, y_train)
# meta_dataset = []
# for i in range(len(num_classes_per_task)):
#     # 生成一些随机数据作为示例
#     x_train = tf.random.normal((100, 2000))  # 100个样本，每个样本2000个特征
#     y_train = tf.random.uniform((100,), maxval=num_classes_per_task[i], dtype=tf.int32)
#     meta_dataset.append((x_train, y_train, classifier_heads[i]))
#
# # 使用MAML训练共享的特征提取器和任务特定的分类头
# maml = MAML(feature_extractor)
# maml.train(meta_dataset, epochs=100)
#
# # 处理新数据
# new_x_train = tf.random.normal((50, 2000))  # 50个新样本
# new_y_train = tf.random.uniform((50,), maxval=5, dtype=tf.int32)  # 新任务有5个类别
#
# # 为新任务生成一个分类头
# new_classifier_head = build_classifier_head(5)
#
# # 将新任务加入meta-dataset并继续训练
# meta_dataset.append((new_x_train, new_y_train, new_classifier_head))
# maml.train(meta_dataset, epochs=20)  # 继续训练20个epoch
#
# def evaluate_accuracy(model, x_test, y_test, classifier_head):
#     features = model(x_test, training=False)
#     predictions = classifier_head(features)
#     predicted_labels = tf.argmax(predictions, axis=1)
#     accuracy = tf.reduce_mean(tf.cast(predicted_labels == y_test, tf.float32))
#     return accuracy.numpy()
#
# # 定义任务的验证集
# x_val_task1 = tf.random.normal((50, 2000))
# y_val_task1 = tf.random.uniform((50,), maxval=11, dtype=tf.int32)
# y_val_task1 = tf.cast(y_val_task1, tf.int64)
# # 继续在meta-dataset上训练，同时评估验证集准确率
# for epoch in range(20):  # 假设再训练20个epoch
#     maml.train(meta_dataset, epochs=1)  # 每次只训练一个epoch
#     acc = evaluate_accuracy(feature_extractor, x_val_task1, y_val_task1, classifier_heads[0])
#     print(f"Epoch {epoch + 1}, Validation Accuracy on Task 1: {acc:.4f}")
import tensorflow as tf
import numpy as np


# 数据标准化函数
def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# 检查数据是否包含 NaN 或 Inf
# def check_data(data_list):
#     for data in data_list:
#         if np.any(np.isnan(data)) or np.any(np.isinf(data)):
#             raise ValueError("Data contains NaN or Inf values!")
def check_data(data_list):
    for data in data_list:
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Data contains NaN or Inf values!")
            return False
    return True

# 特征提取器模型
def build_feature_extractor(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu')
    ])
    return model


# 分类器头，任务特定
def build_classifier_head(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ProtoNet损失函数
def proto_loss(support_embeddings, query_embeddings, support_labels, query_labels, num_classes):
    prototypes = []
    for label in range(num_classes):
        class_embeddings = support_embeddings[support_labels == label]
        prototype = tf.reduce_mean(class_embeddings, axis=0)
        prototypes.append(prototype)

    prototypes = tf.stack(prototypes)
    distances = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=2)
    log_p_y = tf.nn.log_softmax(-distances, axis=1)

    indices = tf.convert_to_tensor([tf.where(query_labels == label)[0][0] for label in query_labels])
    return -tf.reduce_mean(tf.gather(log_p_y, indices, axis=1))


# MAML 类
import tensorflow as tf
import numpy as np
from FTIR_GenerateData import GenerateDate
from sklearn.model_selection import train_test_split


class MAML1:
    def __init__(self, feature_extractor, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1, batch_size=5):
        self.feature_extractor = feature_extractor
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.batch_size = batch_size  # 添加批次大小
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)

    def inner_update(self, x_support, y_support, classifier_head):
        # 内部更新使用支持集
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x_support)
            logits = classifier_head(features)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_support, logits)
        grads = tape.gradient(loss, classifier_head.trainable_variables)
        updated_weights = [w - self.inner_lr * g for w, g in zip(classifier_head.trainable_variables, grads)]
        return updated_weights

    def outer_update(self, meta_batch):
        # 计算当前批次中的外部梯度
        outer_grads = [tf.zeros_like(var) for var in self.feature_extractor.trainable_variables]

        for (x_support, y_support, x_query, y_query, classifier_head) in meta_batch:
            # 执行内更新
            updated_weights = self.inner_update(x_support, y_support, classifier_head)

            # 查询集上的外更新
            with tf.GradientTape() as tape:
                logits = self.model_with_updated_weights(x_query, classifier_head, updated_weights)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_query, logits)
            grads = tape.gradient(loss, self.feature_extractor.trainable_variables)
            outer_grads = [og + g for og, g in zip(outer_grads, grads)]

        # 更新特征提取器的参数
        self.optimizer.apply_gradients(zip(outer_grads, self.feature_extractor.trainable_variables))
    def model_with_updated_weights(self, x, classifier_head, updated_weights):
        temp_model = self.build_temp_model(classifier_head)
        temp_model.set_weights(updated_weights)
        features = self.feature_extractor(x)
        return temp_model(features)

    def build_temp_model(self, classifier_head):
        model_copy = tf.keras.models.clone_model(classifier_head)
        model_copy.set_weights(classifier_head.get_weights())
        return model_copy
    def train(self, meta_dataset, epochs):
        num_tasks = len(meta_dataset)

        for epoch in range(epochs):
            # 随机打乱任务顺序
            np.random.shuffle(meta_dataset)

            # 将任务数据分为小批次
            for i in range(0, num_tasks, self.batch_size):
                meta_batch = meta_dataset[i:i + self.batch_size]
                self.outer_update(meta_batch)

            print(f"Epoch {epoch + 1} completed")
class MAML:
    def __init__(self, feature_extractor, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1):
        self.feature_extractor = feature_extractor
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)

    def inner_update(self, x_support, y_support, classifier_head):
        # 支持集上的内更新
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x_support)  # 使用支持集
            logits = classifier_head(features)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(
                y_support, logits)
        grads = tape.gradient(loss, classifier_head.trainable_variables)
        updated_weights = [w - self.inner_lr * g for w, g in zip(classifier_head.trainable_variables, grads)]
        return updated_weights

    def model_with_updated_weights(self, x, classifier_head, updated_weights):
        temp_model = self.build_temp_model(classifier_head)
        temp_model.set_weights(updated_weights)
        features = self.feature_extractor(x)
        return temp_model(features)

    def build_temp_model(self, classifier_head):
        model_copy = tf.keras.models.clone_model(classifier_head)
        model_copy.set_weights(classifier_head.get_weights())
        return model_copy

    def outer_update(self, meta_dataset):
        outer_grads = [tf.zeros_like(var) for var in self.feature_extractor.trainable_variables]

        for (x_support, y_support, x_query, y_query, classifier_head) in meta_dataset:
            # 支持集上的内更新
            updated_weights = self.inner_update(x_support, y_support, classifier_head)

            # 查询集上的外更新
            with tf.GradientTape() as tape:
                logits = self.model_with_updated_weights(x_query, classifier_head, updated_weights)

                # 调试：打印 logits 和 y_query 的形状
                print(f"logits shape: {logits.shape}, y_query shape: {y_query.shape}")

                # 确保 y_query 的形状与 logits 的形状一致
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(
                    y_query, logits)
            grads = tape.gradient(loss, self.feature_extractor.trainable_variables)
            outer_grads = [og + g for og, g in zip(outer_grads, grads)]

        # 应用外部梯度更新
        self.optimizer.apply_gradients(zip(outer_grads, self.feature_extractor.trainable_variables))

    def train(self, meta_dataset, epochs):
        for epoch in range(epochs):
            self.outer_update(meta_dataset)
            print(f"Epoch {epoch + 1} completed")


# 示例特征提取器和分类头的构建函数
# def build_feature_extractor(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=input_shape),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu')
#     ])
#     return model
def build_feature_extractor(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
    ])
    return model
def build_classifier_head(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 数据标准化函数
def normalize_data(data):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

# 示例数据生成
input_shape = (2000,)
num_classes_per_task = [11, 4, 6]
support_data_list, query_data_list, support_labels_list, query_labels_list = [], [], [], []

# 为每个任务生成支持集和查询集数据
for num_classes in num_classes_per_task:
    x_train = tf.random.normal((100, 2000))  # 100个样本作为支持集
    y_train = tf.random.uniform((100,), maxval=num_classes, dtype=tf.int64)
    x_val = tf.random.normal((50, 2000))  # 50个样本作为查询集
    y_val = tf.random.uniform((50,), maxval=num_classes, dtype=tf.int64)

    support_data_list.append(normalize_data(x_train.numpy()))
    support_labels_list.append(y_train)
    query_data_list.append(normalize_data(x_val.numpy()))
    query_labels_list.append(y_val)

# 特征提取器和分类头
feature_extractor = build_feature_extractor(input_shape)
classifier_heads = [build_classifier_head(num_classes) for num_classes in num_classes_per_task]

# meta-dataset 包含支持集和查询集
meta_dataset = []
for i in range(len(num_classes_per_task)):
    # 每个任务的数据打包成元数据集
    meta_dataset.append((support_data_list[i], support_labels_list[i], query_data_list[i], query_labels_list[i], classifier_heads[i]))


GenData=GenerateDate()
# firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()
firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = GenData.getData()

print(firstData)
print(secondData)
x_train1,y_train1,x_test1,y_test1=GenData.dataAugmenation(firstData,pid1,wavenumber,pname1,1)

x_train3, y_train3, x_test3, y_test3 = GenData.dataAugmenation2(thirdData, pid3, wavenumber, pname3, 1)
# # # fileName='FTIR_PLastics500_c4.csv'
wavenumber4, forthData, pid4, pname4 = GenData.readFromPlastics500('FTIR_PLastics500_c4.csv')
wavenumber5, fifthData, pid5, pname5 = GenData.readFromPlastics500('FTIR_PLastics500_c8.csv')
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

support_data_list = [x_train1, x_train2]

support_labels_list = [y_train1, y_train2]
query_data_list = [x_test1, x_test2]
#
query_labels_list = [y_test1, y_test2]
meta_dataset = []
for i in range(len(num_classes_per_task)):
    # 每个任务的数据打包成元数据集
    meta_dataset.append((support_data_list[i], support_labels_list[i], query_data_list[i], query_labels_list[i], classifier_heads[i]))

# if not check_data(support_data_list) or not check_data(query_data_list):
#     raise ValueError("dadad"
#                      "dada"
#                      "dada"
#                      "dad"
#                      "dada"
#                      "dada"
#                      "Data contains NaN or Inf values")

# 创建 MAML 实例并训练
# maml = MAML1(feature_extractor)
maml = MAML1(feature_extractor, inner_lr=0.001, outer_lr=0.0001, num_inner_steps=5)
maml.train(meta_dataset, epochs=100)

support_data_list = [x_train4]

support_labels_list = [y_train4]
query_data_list = [x_test4]

query_labels_list = [y_test4]
classifier_heads=[build_classifier_head(6)]
# classifier_heads=[6]
# 验证集上的准确率评估
def evaluate_accuracy(model, x_test, y_test, classifier_head):
    features = model(x_test, training=False)
    predictions = classifier_head(features)
    predicted_labels = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(predicted_labels == y_test, tf.float32))
    return accuracy.numpy()


# def fine_tune_and_evaluate(maml, support_set, support_labels, query_set, query_labels, classifier_head):
#     # Step 1: 在支持集上进行fine-tuning
#     updated_weights = maml.inner_update(support_set, support_labels, classifier_head)
#
#     # Step 2: 使用fine-tuning后的模型在查询集上进行预测并计算准确率
#     logits = maml.model_with_updated_weights(query_set, classifier_head, updated_weights)
#     predicted_labels = tf.argmax(logits, axis=1)
#
#     # 计算查询集上的准确率
#     accuracy = tf.reduce_mean(tf.cast(predicted_labels == query_labels, tf.float32))
#     return accuracy.numpy()
def fine_tune_and_evaluate(maml, support_set, support_labels, query_set, query_labels, classifier_head):
    # Step 1: 在支持集上进行fine-tuning
    updated_weights = maml.inner_update(support_set, support_labels, classifier_head)

    # Step 2: 使用fine-tuning后的模型在查询集上进行预测并计算损失
    logits = maml.model_with_updated_weights(query_set, classifier_head, updated_weights)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(query_labels, logits)
    print(f"Query set loss: {loss.numpy()}")

    # 计算准确率
    predicted_labels = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(predicted_labels == query_labels, tf.float32))
    return accuracy.numpy()
for i in range(len(query_data_list)):
    acc = evaluate_accuracy(feature_extractor, query_data_list[i], query_labels_list[i], classifier_heads[i])
    accuracy = fine_tune_and_evaluate(maml, support_data_list[i], support_labels_list[i], query_data_list[i], query_labels_list[i],
                                      classifier_heads[i])
    print(f"Task {i + 1} Validation Accuracy: {acc:.4f}")
    print(f"Task {i + 1} Validation Accuracy2: {accuracy:.4f}")
