# # protonet_ftir.py
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from sklearn.model_selection import train_test_split
#
# # ====== 你的数据加载函数（按你项目实际路径）======
# # get_data(): 返回 firstData, secondData, thirdData, pid1, pid2, pid3, ...
# # readFromPlastics500(): 返回 w, X, y, pName
from FTIR_ReaddataFrom500C4 import readFromPlastics500
# from utils import utils
# # 如果你需要：from your_module import emsc, EMSA  # 保持你现有预处理
#
# # ---------------------------
# # 0) 小工具：按样本最大值归一化（与你现有保持一致即可）
# # ---------------------------
# def max_normalize_rows(X, eps=1e-8):
#     X = np.asarray(X, dtype=np.float32)
#     return X / (np.max(X, axis=1, keepdims=True) + eps)
#
# # ---------------------------
# # 1) Encoder（1D 光谱，L2 归一化）
# # ---------------------------
# def make_encoder(input_len: int, emb_dim: int = 128) -> Model:
#     inp = layers.Input(shape=(input_len,))
#     x = layers.Reshape((input_len, 1))(inp)
#     for f, k in [(64, 9), (64, 9), (128, 7), (128, 7)]:
#         x = layers.Conv1D(f, k, padding="same", use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.ReLU()(x)
#         x = layers.MaxPool1D(2)(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dense(emb_dim, use_bias=False)(x)
#     x = tf.nn.l2_normalize(x, axis=-1)
#     return Model(inp, x, name="encoder")
#
# # ---------------------------
# # 2) Episode 采样（N-way K-shot，Q-query）
# # ---------------------------
# def make_episode(X, y, N=5, K=5, Q=15, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()
#     uniq = np.unique(y)
#     assert len(uniq) >= N, f"类别数不足：{len(uniq)} < N={N}"
#     classes = rng.choice(uniq, size=N, replace=False)
#
#     Sx, Sy, Qx, Qy = [], [], [], []
#     for j, c in enumerate(classes):
#         idx = np.where(y == c)[0]
#         idx = rng.permutation(idx)
#         assert len(idx) >= K + Q, f"类 {c} 样本不足 K+Q={K+Q}"
#         s, q = idx[:K], idx[K:K+Q]
#         Sx.append(X[s]); Qx.append(X[q])
#         Sy.extend([j]*len(s)); Qy.extend([j]*len(q))
#
#     Sx = np.concatenate(Sx, axis=0)
#     Qx = np.concatenate(Qx, axis=0)
#     Sy = np.asarray(Sy, dtype=np.int32)  # 0..N-1
#     Qy = np.asarray(Qy, dtype=np.int32)  # 0..N-1
#     return Sx, Sy, Qx, Qy, classes  # classes 保留原始类ID映射
#
# # ---------------------------
# # 3) 原型 + 对角马氏参数（张量实现）
# # ---------------------------
# @tf.function
# def build_prototypes(emb_s, y_s, N):
#     y_s = tf.cast(y_s, tf.int32)
#     N   = tf.cast(N,  tf.int32)
#     protos = tf.math.unsorted_segment_mean(emb_s, y_s, num_segments=N)  # [N,D]
#     return protos
#
# @tf.function
# def build_mahalanobis_diag_params(emb_s, y_s, N, eps=1e-3):
#     y_s = tf.cast(y_s, tf.int32)
#     N   = tf.cast(N,  tf.int32)
#     protos = build_prototypes(emb_s, y_s, N)  # [N,D]
#
#     proto_for_each = tf.gather(protos, y_s)   # [NK,D]
#     diff = emb_s - proto_for_each             # [NK,D]
#     var  = tf.math.unsorted_segment_mean(tf.square(diff), y_s, num_segments=N)  # [N,D]
#     inv_vars = 1.0 / (var + eps)  # 稳定化
#     return protos, inv_vars
#
# # ---------------------------
# # 4) Proto 损失（欧氏 / 对角马氏）
# # ---------------------------
# @tf.function
# def prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy, distance="euclid", temperature=1.0):
#     Sx = tf.cast(Sx, tf.float32)
#     Qx = tf.cast(Qx, tf.float32)
#     Sy = tf.cast(Sy, tf.int32)
#     Qy = tf.cast(Qy, tf.int32)
#
#     emb_s = encoder(Sx, training=True)  # [NK,D]
#     emb_q = encoder(Qx, training=True)  # [NQ,D]
#     N = tf.reduce_max(Sy) + 1
#
#     if distance == "euclid":
#         protos = build_prototypes(emb_s, Sy, N)                      # [N,D]
#         diff   = tf.expand_dims(emb_q, 1) - tf.expand_dims(protos, 0)  # [NQ,N,D]
#         dist2  = tf.reduce_sum(tf.square(diff), axis=-1)             # [NQ,N]
#         logits = -dist2 / temperature
#     elif distance == "maha_diag":
#         protos, inv_vars = build_mahalanobis_diag_params(emb_s, Sy, N)  # [N,D], [N,D]
#         diff   = tf.expand_dims(emb_q, 1) - tf.expand_dims(protos, 0)   # [NQ,N,D]
#         dist2  = tf.reduce_sum(tf.square(diff) * tf.expand_dims(inv_vars, 0), axis=-1)  # [NQ,N]
#         logits = -dist2 / temperature
#     else:
#         raise ValueError("distance must be 'euclid' or 'maha_diag'")
#
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Qy, logits=logits))
#     pred = tf.argmax(logits, axis=-1, output_type=Qy.dtype)
#     acc  = tf.reduce_mean(tf.cast(tf.equal(pred, Qy), tf.float32))
#     return loss, acc
#
# # ---------------------------
# # 5) 训练循环（episodic）
# # ---------------------------
# def train_protonet(encoder: Model, X_tr, y_tr, X_val, y_val,
#                    steps=2000, N=5, K=5, Q=15, lr=1e-3, seed=0,
#                    val_every=100, val_episodes=50, distance="euclid", temperature=1.0):
#     rng = np.random.default_rng(seed)
#     opt = tf.keras.optimizers.Adam(lr)
#     best_val = -1.0
#     best_weights = encoder.get_weights()
#
#     @tf.function
#     def _train_step(Sx, Sy, Qx, Qy):
#         with tf.GradientTape() as tape:
#             loss, acc = prototypical_loss_and_acc(
#                 encoder, Sx, Sy, Qx, Qy,
#                 distance=distance, temperature=temperature
#             )
#         grads = tape.gradient(loss, encoder.trainable_variables)
#         opt.apply_gradients(zip(grads, encoder.trainable_variables))
#         return loss, acc
#
#     for t in range(1, steps+1):
#         Sx, Sy, Qx, Qy, _ = make_episode(X_tr, y_tr, N, K, Q, rng)
#         loss, acc = _train_step(Sx, Sy, Qx, Qy)
#
#         if t % val_every == 0:
#             accs = []
#             for _ in range(val_episodes):
#                 Sxv, Syv, Qxv, Qyv, _ = make_episode(X_val, y_val, N, K, Q, rng)
#                 _, a = prototypical_loss_and_acc(
#                     encoder, Sxv, Syv, Qxv, Qyv,
#                     distance=distance, temperature=temperature
#                 )
#                 accs.append(float(a))
#             val_acc = float(np.mean(accs))
#             if val_acc > best_val:
#                 best_val = val_acc
#                 best_weights = encoder.get_weights()
#             print(f"[step {t}] train_acc={float(acc):.3f}  val_acc={val_acc:.3f}")
#
#     encoder.set_weights(best_weights)
#     print(f"Loaded best encoder (val_acc={best_val:.3f})")
#     return encoder
#
# # ---------------------------
# # 6) 评测（episodic）
# # ---------------------------
# def predict_episode(encoder: Model, Sx, Sy, Qx, classes, distance="euclid", eps=1e-3):
#     from scipy.spatial.distance import cdist
#     emb_s = encoder.predict(Sx, batch_size=256, verbose=0)
#     emb_q = encoder.predict(Qx, batch_size=256, verbose=0)
#     N = len(np.unique(Sy))
#     Sy = Sy.astype(np.int32)
#
#     if distance == "euclid":
#         protos = np.stack([emb_s[Sy==c].mean(axis=0) for c in range(N)], axis=0)
#         D2 = cdist(emb_q, protos, metric="sqeuclidean")
#     elif distance == "maha_diag":
#         protos = np.stack([emb_s[Sy==c].mean(axis=0) for c in range(N)], axis=0)
#         vars_  = np.stack([emb_s[Sy==c].var(axis=0)  for c in range(N)], axis=0)
#         invv   = 1.0 / (vars_ + eps)
#         diff   = emb_q[:, None, :] - protos[None, :, :]
#         D2     = np.sum(diff*diff * invv[None, :, :], axis=-1)
#     else:
#         raise ValueError("distance must be 'euclid' or 'maha_diag'")
#
#     pred_idx = np.argmin(D2, axis=1)
#     return classes[pred_idx]
#
# def episodic_eval(encoder: Model, X, y, episodes=200, N=5, K=5, Q=15, seed=0, distance="euclid"):
#     rng = np.random.default_rng(seed)
#     correct = 0; total = 0
#     for _ in range(episodes):
#         Sx, Sy, Qx, Qy, classes = make_episode(X, y, N, K, Q, rng)
#         pred_labels = predict_episode(encoder, Sx, Sy, Qx, classes, distance=distance)
#         true_labels = classes[Qy]
#         correct += (pred_labels == true_labels).sum()
#         total   += len(true_labels)
#     return correct / total
#
# # ---------------------------
# # 7) main：c8 训练 -> c8/c4 评测（度量可选）
# # ---------------------------
# def main():
#     # === 载入你的数据 ===
#     # 你已有 get_data()，但这里只演示 c4/c8 两套（可按需扩展）
#     w4,  X4,  y4,  p4 = readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')
#     w8,  X8,  y8,  p8 = readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
#
#     # === 保持与你实验一致的预处理（例如最大值归一化/EMSC/EMSA 等）===
#     # X4 = max_normalize_rows(X4)
#     # X8 = max_normalize_rows(X8)
#     # 若你要插值/EMSC/EMSA，请在这里插入；务必训练/评测一致
#
#     # === c8 train/val 划分并训练 ===
#     X_tr, X_val, y_tr, y_val = train_test_split(X8, y8, test_size=0.2, stratify=y8, random_state=42)
#     encoder = make_encoder(input_len=X_tr.shape[1], emb_dim=128)
#
#     # 切换距离：'euclid' 或 'maha_diag'
#     distance = "maha_diag"
#     encoder = train_protonet(
#         encoder, X_tr, y_tr, X_val, y_val,
#         steps=500, N=5, K=5, Q=15, lr=1e-3, seed=0,
#         val_every=100, val_episodes=50,
#         distance=distance, temperature=1.0
#     )
#
#     # === 同域 episodic 评测（c8 验证集）===
#     acc_in = episodic_eval(encoder, X_val, y_val, episodes=200, N=5, K=5, Q=15, seed=1, distance=distance)
#     print(f"[c8 同域 episodic acc] {acc_in:.4f}")
#
#     # === 跨域 episodic 评测（c4）===
#     acc_cross = episodic_eval(encoder, X4, y4, episodes=200, N=5, K=5, Q=15, seed=2, distance=distance)
#     print(f"[c8 训练 -> c4 评测 episodic acc] {acc_cross:.4f}")
#
# if __name__ == "__main__":
#     # 可选：GPU 显存按需增长
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except Exception as e:
#             print(e)
#     main()
# -*- coding: utf-8 -*-
"""
ProtoNet + Class-wise Mahalanobis (diag/full) for 1D FTIR spectra
- Episodic training (N-way, K-shot, Q-query)
- LayerNorm + L2 normalized embedding
- Works with your FTIR loaders if available; otherwise falls back to synthetic data.
"""
# -*- coding: utf-8 -*-
"""
Prototypical Networks (1D) with class-wise Mahalanobis (diag/full).
- Episodic training on source dataset (e.g., c8)
- Episodic evaluation on source (in-domain) and target (cross-domain)
Author: Xinyu + ChatGPT
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

# ==============
# 0) 小设置
# ==============
def setup_tf_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)

setup_tf_memory_growth()
tf.random.set_seed(42)
np.random.seed(42)


# ==============
# 1) Encoder（1D 光谱，输出 L2 归一化 embedding）
# ==============
def make_encoder(input_len: int, emb_dim: int = 128) -> Model:
    inp = layers.Input(shape=(input_len,))
    x = layers.Reshape((input_len, 1))(inp)
    # 轻量 1D CNN backbone
    for f, k in [(64, 9), (64, 9), (128, 7), (128, 7)]:
        x = layers.Conv1D(f, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(emb_dim, use_bias=False)(x)
    x = tf.nn.l2_normalize(x, axis=-1)  # L2 normalize
    return Model(inp, x, name="encoder")


# ==============
# 2) 采样一个 episode（N way, K shot, Q query）
#    返回支持集/查询集 + 0..N-1 的内部标签 + 原始类ID映射
# ==============
def make_episode(X, y, N=5, K=5, Q=15, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    uniq = np.unique(y)
    assert len(uniq) >= N, f"类别数不足：{len(uniq)} < N={N}"
    classes = rng.choice(uniq, size=N, replace=False)

    Sx, Sy, Qx, Qy = [], [], [], []
    for j, c in enumerate(classes):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        assert len(idx) >= K + Q, f"类 {c} 样本不足 K+Q={K+Q}"
        s, q = idx[:K], idx[K:K + Q]
        Sx.append(X[s]); Qx.append(X[q])
        Sy.extend([j] * len(s)); Qy.extend([j] * len(q))

    Sx = np.concatenate(Sx, axis=0)
    Qx = np.concatenate(Qx, axis=0)
    Sy = np.asarray(Sy, dtype=np.int32)
    Qy = np.asarray(Qy, dtype=np.int32)
    return Sx, Sy, Qx, Qy, classes


# ==============
# 3) 各种“原型/协方差/距离”构建
# ==============

def build_prototypes(emb_s, Sy, N):
    """每类均值（原型） [N, D]"""
    Sy = tf.cast(Sy, tf.int32)
    N  = tf.cast(N,  tf.int32)
    protos = tf.math.unsorted_segment_mean(emb_s, Sy, num_segments=N)  # [N, D]
    return protos

def class_stats_diag(emb_s, Sy, N, eps=1e-3):
    """
    每类对角协方差（返回 inv_var）
    emb_s: [NK, D]
    Sy:    [NK]
    返回:
      mu:     [N, D]
      invvar: [N, D]
    """
    mu = build_prototypes(emb_s, Sy, N)  # [N,D]
    mu_per_sample = tf.gather(mu, tf.cast(Sy, tf.int32))          # [NK, D]
    xc = emb_s - mu_per_sample                                    # [NK, D]
    var_sum = tf.math.unsorted_segment_sum(tf.square(xc), tf.cast(Sy, tf.int32), num_segments=tf.cast(N, tf.int32))  # [N,D]
    cnt = tf.math.unsorted_segment_sum(tf.ones_like(xc), tf.cast(Sy, tf.int32), num_segments=tf.cast(N, tf.int32))   # [N,D]
    var = var_sum / tf.maximum(cnt, 1.0)                          # [N, D]
    invvar = 1.0 / (var + eps)
    return mu, invvar

def class_stats_full(emb_s, Sy, N, eps=1e-3):
    """
    每类完整协方差（返回协方差逆）
    emb_s: [NK, D]
    Sy:    [NK]
    返回:
      mu:      [N, D]
      inv_cov: [N, D, D]
    """
    Sy = tf.cast(Sy, tf.int32)
    N  = tf.cast(N,  tf.int32)
    mu = tf.math.unsorted_segment_mean(emb_s, Sy, num_segments=N)  # [N,D]
    D  = tf.shape(emb_s)[1]

    mu_per_sample = tf.gather(mu, Sy)             # [NK, D]
    xc = emb_s - mu_per_sample                    # [NK, D]
    outer = tf.einsum('bi,bj->bij', xc, xc)       # [NK, D, D]
    cov_sum = tf.math.unsorted_segment_sum(outer, Sy, num_segments=N)  # [N, D, D]

    ones = tf.ones((tf.shape(emb_s)[0], 1), dtype=emb_s.dtype)
    cnt  = tf.math.unsorted_segment_sum(ones, Sy, num_segments=N)[:, 0]  # [N]
    cnt  = tf.maximum(cnt, 1.0)
    cov  = cov_sum / tf.reshape(cnt, [-1, 1, 1])  # [N, D, D]

    I = tf.eye(D, dtype=emb_s.dtype)[None, :, :]
    inv_cov = tf.linalg.inv(cov + eps * I)        # [N, D, D]
    return mu, inv_cov

def dists_euclid(emb_q, protos):
    """欧氏距离平方 [BQ, N]"""
    diff = tf.expand_dims(emb_q, 1) - tf.expand_dims(protos, 0)  # [BQ,N,D]
    d2 = tf.reduce_sum(tf.square(diff), axis=-1)                 # [BQ,N]
    return d2

def dists_maha_diag(emb_q, mu, invvar):
    """每类对角马氏距离 [BQ, N]"""
    diff = tf.expand_dims(emb_q, 1) - tf.expand_dims(mu, 0)      # [BQ,N,D]
    d2 = tf.reduce_sum(tf.square(diff) * tf.expand_dims(invvar, 0), axis=-1)  # [BQ,N]
    return d2

def dists_maha_full(emb_q, mu, inv_cov):
    """每类完整马氏距离 [BQ, N]"""
    diff = tf.expand_dims(emb_q, 1) - tf.expand_dims(mu, 0)      # [BQ,N,D]
    Av  = tf.einsum('ndd,bnd->bnd', inv_cov, diff)               # [BQ,N,D]
    d2  = tf.reduce_sum(diff * Av, axis=-1)                      # [BQ,N]
    return d2


# ==============
# 4) 原型网络的 loss + acc（支持三种距离）
# ==============
@tf.function
def prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy, distance='euclid'):
    Sx = tf.cast(Sx, tf.float32)
    Qx = tf.cast(Qx, tf.float32)
    Sy = tf.cast(Sy, tf.int32)
    Qy = tf.cast(Qy, tf.int32)

    emb_s = encoder(Sx, training=True)   # [N*K, D]
    emb_q = encoder(Qx, training=True)   # [N*Q, D]
    N = tf.reduce_max(Sy) + 1            # 标量 int32

    if distance == 'euclid':
        protos = build_prototypes(emb_s, Sy, N)                  # [N, D]
        d2 = dists_euclid(emb_q, protos)                         # [BQ,N]
    elif distance == 'maha_diag':
        mu, invvar = class_stats_diag(emb_s, Sy, N)              # [N,D], [N,D]
        d2 = dists_maha_diag(emb_q, mu, invvar)                  # [BQ,N]
    elif distance == 'maha_full':
        mu, inv_cov = class_stats_full(emb_s, Sy, N)             # [N,D], [N,D,D]
        d2 = dists_maha_full(emb_q, mu, inv_cov)                 # [BQ,N]
    else:
        raise ValueError("distance must be one of: 'euclid', 'maha_diag', 'maha_full'")

    logits = -d2
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Qy, logits=logits))
    pred = tf.argmax(logits, axis=-1, output_type=Qy.dtype)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, Qy), tf.float32))
    return loss, acc


# ==============
# 5) 训练循环（episodic）
# ==============
def train_protonet(encoder: Model, X_tr, y_tr, X_val, y_val,
                   steps=2000, N=5, K=5, Q=15, lr=1e-3, seed=0,
                   val_every=100, val_episodes=50, distance='euclid'):
    rng = np.random.default_rng(seed)
    opt = tf.keras.optimizers.Adam(lr)
    best_val = -1.0
    best_weights = encoder.get_weights()

    @tf.function
    def _train_step(Sx, Sy, Qx, Qy):
        with tf.GradientTape() as tape:
            loss, acc = prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy, distance=distance)
        grads = tape.gradient(loss, encoder.trainable_variables)
        opt.apply_gradients(zip(grads, encoder.trainable_variables))
        return loss, acc

    for t in range(1, steps + 1):
        Sx, Sy, Qx, Qy, _ = make_episode(X_tr, y_tr, N, K, Q, rng)
        loss, acc = _train_step(Sx, Sy, Qx, Qy)

        if t % val_every == 0:
            accs = []
            for _ in range(val_episodes):
                Sxv, Syv, Qxv, Qyv, _ = make_episode(X_val, y_val, N, K, Q, rng)
                _, a = prototypical_loss_and_acc(encoder, Sxv, Syv, Qxv, Qyv, distance=distance)
                accs.append(float(a))
            val_acc = float(np.mean(accs))
            if val_acc > best_val:
                best_val = val_acc
                best_weights = encoder.get_weights()
            print(f"[step {t}] train_acc={float(acc):.3f}  val_acc={val_acc:.3f}")

    encoder.set_weights(best_weights)
    print(f"Loaded best encoder (val_acc={best_val:.3f})")
    return encoder


# ==============
# 6) Episodic 评测（支持三种距离）
# ==============
def episodic_eval(encoder: Model, X, y, episodes=200, N=5, K=5, Q=15, seed=0, distance='euclid'):
    rng = np.random.default_rng(seed)
    total = 0
    correct = 0
    for _ in range(episodes):
        Sx, Sy, Qx, Qy, classes = make_episode(X, y, N, K, Q, rng)
        # 前向（推理时不必计算梯度）
        Sx_tf = tf.convert_to_tensor(Sx, tf.float32)
        Qx_tf = tf.convert_to_tensor(Qx, tf.float32)
        Sy_tf = tf.convert_to_tensor(Sy, tf.int32)
        Qy_tf = tf.convert_to_tensor(Qy, tf.int32)
        emb_s = encoder(Sx_tf, training=False)
        emb_q = encoder(Qx_tf, training=False)
        N_tf = tf.reduce_max(Sy_tf) + 1

        if distance == 'euclid':
            protos = build_prototypes(emb_s, Sy_tf, N_tf)
            d2 = dists_euclid(emb_q, protos).numpy()
        elif distance == 'maha_diag':
            mu, invvar = class_stats_diag(emb_s, Sy_tf, N_tf)
            d2 = dists_maha_diag(emb_q, mu, invvar).numpy()
        elif distance == 'maha_full':
            mu, inv_cov = class_stats_full(emb_s, Sy_tf, N_tf)
            d2 = dists_maha_full(emb_q, mu, inv_cov).numpy()
        else:
            raise ValueError("distance must be one of: 'euclid', 'maha_diag', 'maha_full'")

        pred_idx = d2.argmin(axis=1)      # 0..N-1
        pred_labels = classes[pred_idx]    # 映射回原始类ID
        true_labels = classes[Qy]          # 同样映射
        correct += (pred_labels == true_labels).sum()
        total += len(true_labels)
    return correct / total


# ==============
# 7) 单 episode 推理（返回标签），支持三种距离
# ==============
def predict_episode(encoder: Model, Sx, Sy, Qx, classes, distance='euclid', eps=1e-3):
    Sx_tf = tf.convert_to_tensor(Sx, tf.float32)
    Qx_tf = tf.convert_to_tensor(Qx, tf.float32)
    Sy_tf = tf.convert_to_tensor(Sy, tf.int32)
    emb_s = encoder(Sx_tf, training=False)
    emb_q = encoder(Qx_tf, training=False)
    N_tf = tf.reduce_max(Sy_tf) + 1

    if distance == 'euclid':
        protos = build_prototypes(emb_s, Sy_tf, N_tf)                    # [N,D]
        d2 = dists_euclid(emb_q, protos).numpy()                         # [BQ,N]
    elif distance == 'maha_diag':
        mu, invvar = class_stats_diag(emb_s, Sy_tf, N_tf)                # [N,D], [N,D]
        d2 = dists_maha_diag(emb_q, mu, invvar).numpy()
    elif distance == 'maha_full':
        mu, inv_cov = class_stats_full(emb_s, Sy_tf, N_tf)               # [N,D], [N,D,D]
        d2 = dists_maha_full(emb_q, mu, inv_cov).numpy()
    else:
        raise ValueError("distance must be one of: 'euclid', 'maha_diag', 'maha_full'")

    pred_idx = np.argmin(d2, axis=1)
    return classes[pred_idx]


# ==============
# 8) 一个 main 示例（请替换成你的数据 X_src/y_src 和 X_tgt/y_tgt）
# ==============
def main():
    # ====== 准备你的数据 ======
    # 这里用随机数据示意：请替换为你的真实数据（例如 c8 -> c4）
    # X_* 形状 [num_samples, L]；y_* 为整数标签（不必从0开始）
    w5, X5, y5, p5 = readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
    w4, X4, y4, p4 = readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')

    # 若你已经做了插值/EMSC/EMSA统一到同一波数轴，这里直接使用处理后的 X5/X4
    # 归一化（与训练/测试保持一致；如果你有固定策略，请保持一致）
    X_src, y_src = X5, y5
    X_tgt, y_tgt = X4, y4

    # 切分源域 train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_src, y_src, test_size=0.3, random_state=42, stratify=y_src)

    # ====== 构建 encoder ======
    encoder = make_encoder(input_len=X_src.shape[1], emb_dim=128)

    # 选择距离：'euclid' / 'maha_diag' / 'maha_full'
    distance = 'maha_full'  # 你想要的“每类完整协方差马氏距离”

    # ====== episodic 训练（在源域）======
    encoder = train_protonet(
        encoder, X_tr, y_tr, X_val, y_val,
        steps=500, N=6, K=5, Q=15, lr=1e-3, seed=0,
        val_every=100, val_episodes=30, distance=distance
    )

    # ====== 源域 episodic 评测 ======
    acc_in = episodic_eval(encoder, X_val, y_val, episodes=100, N=6, K=5, Q=15, seed=1, distance=distance)
    print(f"[源域 episodic acc] {acc_in:.4f}")

    # ====== 跨域 episodic 评测（在目标域）======
    acc_cross = episodic_eval(encoder, X_tgt, y_tgt, episodes=100, N=6, K=5, Q=15, seed=2, distance=distance)
    print(f"[跨域 episodic acc] {acc_cross:.4f}")

    # ====== 单 episode 推理示例 ======
    Sx, Sy, Qx, Qy, classes = make_episode(X_tgt, y_tgt, N=6, K=5, Q=10, rng=np.random.default_rng(3))
    pred = predict_episode(encoder, Sx, Sy, Qx, classes, distance=distance)
    true_labels = classes[Qy]
    print(f"[单 episode acc] {(pred == true_labels).mean():.4f} ({distance})")


if __name__ == "__main__":
    main()
