import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from FTIR_GenerateData import GenerateDate

######################################
# 1. 定义预训练函数
######################################
def pretrain_feature_extractor(feature_extractor,
                               x_train, y_train,
                               x_val, y_val,
                               num_classes,
                               epochs=10,
                               batch_size=32,
                               lr=1e-3):
    """
    使用常规的监督学习方式，在单个数据集上预训练特征提取器和一个分类头。
    """
    classifier_head = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    features = feature_extractor(inputs)
    outputs = classifier_head(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=1000) \
        .batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    for epoch in range(epochs):
        # ----- 训练阶段 -----
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc_metric.update_state(y_batch, logits)

        train_acc = train_acc_metric.result().numpy()
        train_acc_metric.reset_states()

        # ----- 验证阶段 -----
        for x_batch_val, y_batch_val in val_ds:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result().numpy()
        val_acc_metric.reset_states()

        print(f"Pretrain Epoch {epoch + 1}/{epochs}: "
              f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    return classifier_head


######################################
# 2. 修改 MAML 类
######################################
class MAML:
    def __init__(self, feature_extractor, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1):
        self.feature_extractor = feature_extractor
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps

        # 使用 Adam 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)

    def inner_update(self, x_support, y_support, classifier_head):
        """
        在支持集上进行几步梯度更新，只更新分类头权重，或（可选）也更新特征提取器。
        """
        updated_weights = classifier_head.get_weights()

        # 执行 num_inner_steps 步
        for step in range(self.num_inner_steps):
            with tf.GradientTape() as tape:
                # 注意，如果想一起更新特征提取器，需要把 feature_extractor.trainable_variables 也加进来
                features = self.feature_extractor(x_support, training=True)
                logits = self._forward_with_weights(features, classifier_head, updated_weights)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
                    y_support, logits)

            # 只对分类头做梯度
            grads = tape.gradient(loss, [tf.Variable(w) for w in updated_weights])
            updated_weights = [w - self.inner_lr*g for w, g in zip(updated_weights, grads)]

        return updated_weights

    def model_with_updated_weights(self, x, classifier_head, updated_weights):
        features = self.feature_extractor(x, training=False)
        logits = self._forward_with_weights(features, classifier_head, updated_weights)
        return logits

    def _forward_with_weights(self, features, classifier_head, weights):
        """
        自定义一个前向传播，用给定的权重 weights 来计算输出。
        """
        # 建一个临时模型并赋值
        temp_model = tf.keras.models.clone_model(classifier_head)
        temp_model.set_weights(weights)
        return temp_model(features)

    def outer_update(self, meta_dataset):
        """
        对特征提取器的梯度累加并更新。
        如果你想冻结特征提取器，则无需对其计算梯度；反之，如果要更新它，
        则这里会需要 tape.watch(...) 并做相应处理。
        """
        # 如果要冻结特征提取器，这里就不需要累加了，
        # 直接跳过对 self.feature_extractor 的更新。
        # 若需要更新特征提取器，则 outer_grads = [tf.zeros_like(var) for var in self.feature_extractor.trainable_variables]
        outer_grads = [tf.zeros_like(var) for var in self.feature_extractor.trainable_variables]

        for (x_support, y_support, x_query, y_query, classifier_head) in meta_dataset:
            # 1) 内更新
            updated_weights = self.inner_update(x_support, y_support, classifier_head)

            # 2) 外更新(在查询集上算loss)
            #   如果冻结特征提取器，那就不需要再对 feature_extractor 求梯度了。
            #   这里给出“如果需要更新特征提取器”时的写法。
            with tf.GradientTape() as tape:
                tape.watch(self.feature_extractor.trainable_variables)
                logits = self.model_with_updated_weights(x_query, classifier_head, updated_weights)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
                    y_query, logits)

            # 打印 loss 以便观察
            print(f"[Debug] Query Loss: {loss.numpy():.4f}")

            # 计算并累加梯度
            grads = tape.gradient(loss, self.feature_extractor.trainable_variables)
            outer_grads = [og + g for og, g in zip(outer_grads, grads)]

        # 在这里更新特征提取器
        self.optimizer.apply_gradients(zip(outer_grads, self.feature_extractor.trainable_variables))

    def train(self, meta_dataset, epochs):
        for epoch in range(epochs):
            print(f"\n=== [MAML] Epoch {epoch + 1}/{epochs} ===")
            self.outer_update(meta_dataset)


######################################
# 3. 在 main 函数中做相应修改
######################################
if __name__ == "__main__":
    # ============ (A) 先做预训练阶段 ============
    # 假设我们先在一个“单任务”数据集上进行普通训练，这里随便模拟
    x_all = np.random.randn(500, 2000).astype(np.float32)  # 500 条样本，特征维度 2000
    y_all = np.random.randint(0, 10, (500,)).astype(np.int32)  # 假设有 10 个类别

    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=123)
    GenData = GenerateDate()
    # firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()
    firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = GenData.getData()

    print(firstData)
    print(secondData)
    x_train1, y_train1, x_test1, y_test1 = GenData.dataAugmenation(firstData, pid1, wavenumber, pname1, 1)

    x_train3, y_train3, x_test3, y_test3 = GenData.dataAugmenation2(thirdData, pid3, wavenumber, pname3, 1)
    # # # fileName='FTIR_PLastics500_c4.csv'
    wavenumber4, forthData, pid4, pname4 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')
    wavenumber5, fifthData, pid5, pname5 = GenData.readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
    # #x_train3, x_test3, y_train3, y_test3 = train_test_split(thirdData, pid3, test_size=0.3, random_state=1)
    # # #x_train1, x_test1, y_train1, y_test1 = train_test_split(firstData, pid1, test_size=0.3, random_state=1)
    x_train2, y_train2, x_test2, y_test2 = GenData.dataAugmenation2(secondData, pid2, wavenumber, pname2, 1)
    # # x_train4, y_train4, x_test4, y_test4 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
    # x_train2, x_test2, y_train2, y_test2 = train_test_split(secondData, pid2, test_size=0.3, random_state=1)
    # print(forthData.shape)
    # x_train1, y_train1, x_test1, y_test1 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
    x_train4, x_test4, y_train4, y_test4 = train_test_split(forthData, pid4, test_size=0.7, random_state=1)
    x_train5, x_test5, y_train5, y_test5 = train_test_split(fifthData, pid5, test_size=0.7, random_state=1)
    for item in x_train1:
        if np.any(np.isnan(item)) or np.any(np.isinf(item)):
            print('x_train1', item)

    for item in x_train2:
        if np.any(np.isnan(item)) or np.any(np.isinf(item)):
            print('x_train2', item)
    input_shape = 2000
    num_tasks = 3
    num_classes_per_task = [11, 4]

    # 构建特征提取器
    def build_feature_extractor(input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')
        ])
        return model


    feature_extractor = build_feature_extractor((2000,))

    # 先做预训练
    print("\n===== Start Pretraining Feature Extractor =====")
    pretrain_head = pretrain_feature_extractor(
        feature_extractor=feature_extractor,
        x_train=x_train1,
        y_train=y_train1,
        x_val=x_test1,
        y_val=y_test1,
        num_classes=11,
        epochs=5,  # 这里少训几轮演示，实际可多训
        batch_size=32,
        lr=1e-3
    )

    # ============ (B) 再做 MAML 阶段 ============
    for layer in feature_extractor.layers:
        layer.trainable = False

    num_classes_per_task = [6, 6]

    meta_dataset = []
    # for i, num_classes in enumerate(num_classes_per_task):
        # 随机生成“支持集”“查询集”
        # x_support = np.random.randn(100, 2000).astype(np.float32)
        # y_support = np.random.randint(0, num_classes, size=(100,))
        # x_query = np.random.randn(50, 2000).astype(np.float32)
        # y_query = np.random.randint(0, num_classes, size=(50,))

        # 该任务对应的分类头
        # classifier_head= tf.keras.Sequential([
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])



        # 打包进元数据集中
    # meta_dataset.append((x_train4, y_train4, x_test4, y_test4, classifier_head))
    # 假设你的输入特征维度是 2000
    feature_dim = 2000

    # 创建分类头
    classifier_head1 = tf.keras.Sequential([
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    # # 显式 build
    # classifier_head1.build(input_shape=(None, 64))
    # classifier_head2 = tf.keras.Sequential([
    #     tf.keras.layers.Dense(6, activation='softmax')
    # ])
    # # 显式 build
    # classifier_head2.build(input_shape=(None, 64))
    #meta_dataset.append((x_train5, y_train5, x_test5, y_test5, classifier_head))
    meta_dataset.append((x_train4, y_train4, x_test4, y_test4, classifier_head1))
    # 创建 MAML 实例，用“预训练后的” feature_extractor 作为初始状态
    maml = MAML(feature_extractor, inner_lr=1e-3, outer_lr=1e-4, num_inner_steps=1)
    print("\n===== Start MAML Meta-Training =====")
    maml.train(meta_dataset, epochs=20)


    def fine_tune_and_evaluate(maml_obj, x_support, y_support, x_query, y_query, classifier_head):
        """
        在支持集上进行若干步 inner update，然后在查询集上评估精度。
        """
        # 做 inner update
        updated_weights = maml_obj.inner_update(x_support, y_support, classifier_head)
        # 用更新后的权重做推理
        logits = maml_obj.model_with_updated_weights(x_query, classifier_head, updated_weights)
        preds = tf.argmax(logits, axis=1)
        acc = tf.reduce_mean(tf.cast(preds == y_query, tf.float32))
        return acc.numpy()


    # 演示对 meta_dataset[0] 做评估
    (x_support, y_support, x_query, y_query, classifier_head) = meta_dataset[0]
    acc_0 = fine_tune_and_evaluate(maml, x_support, y_support, x_query, y_query, classifier_head)
    print(f"After MAML, Task 1 accuracy = {acc_0:.4f}")
