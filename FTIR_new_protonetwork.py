import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.spatial.distance import cdist
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense
from scipy import interpolate
from utils import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Union as U, Tuple as T
from FTIR_ReaddataFrom500C4 import readFromPlastics500
from utils import utils

from sklearn.model_selection import train_test_split
# =========================
# 1) Encoder（1D 光谱，L2 归一化）
# =========================
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
    # L2 normalize
    x = tf.nn.l2_normalize(x, axis=-1)
    return Model(inp, x, name="encoder")


# =========================
# 2) Episode 采样（N-way K-shot，Q-query）
#    返回 episode 内部 0..N-1 的标签
# =========================
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
        s, q = idx[:K], idx[K:K+Q]
        Sx.append(X[s]); Qx.append(X[q])
        Sy.extend([j]*len(s)); Qy.extend([j]*len(q))

    Sx = np.concatenate(Sx, axis=0)
    Qx = np.concatenate(Qx, axis=0)
    Sy = np.asarray(Sy, dtype=np.int32)
    Qy = np.asarray(Qy, dtype=np.int32)
    return Sx, Sy, Qx, Qy, classes  # 注意：classes 保存原始类ID顺序


# =========================
# 3) Proto 构建与 logits 计算
# =========================
def build_prototypes(emb_support, y_support, N):
    # emb_support: [NK, D] float32
    # y_support:  [NK]    int32, 值域 0..N-1
    y_support = tf.cast(y_support, tf.int32)
    N = tf.cast(N, tf.int32)
    protos = tf.math.unsorted_segment_mean(emb_support, y_support, num_segments=N)  # [N, D]
    return protos



@tf.function
@tf.function
def prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy):
    Sx = tf.cast(Sx, tf.float32)
    Qx = tf.cast(Qx, tf.float32)
    Sy = tf.cast(Sy, tf.int32)
    Qy = tf.cast(Qy, tf.int32)

    emb_s = encoder(Sx, training=True)   # [N*K, D]
    emb_q = encoder(Qx, training=True)   # [N*Q, D]
    N = tf.reduce_max(Sy) + 1            # 标量张量 int32

    protos = build_prototypes(emb_s, Sy, N)  # [N, D]

    diff = tf.expand_dims(emb_q, 1) - tf.expand_dims(protos, 0)  # [BQ,N,D]
    dist2 = tf.reduce_sum(tf.square(diff), axis=-1)              # [BQ,N]
    logits = -dist2
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Qy, logits=logits)
    )
    pred = tf.argmax(logits, axis=-1, output_type=Qy.dtype)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, Qy), tf.float32))
    return loss, acc



# =========================
# 4) 训练循环（episodic）
# =========================
def train_protonet(encoder: Model, X_tr, y_tr, X_val, y_val,
                   steps=2000, N=5, K=5, Q=15, lr=1e-3, seed=0, val_every=100, val_episodes=50):
    rng = np.random.default_rng(seed)
    opt = tf.keras.optimizers.Adam(lr)
    best_val = -1.0
    best_weights = encoder.get_weights()

    @tf.function
    @tf.function
    def _train_step(Sx, Sy, Qx, Qy):
        Sx = tf.cast(Sx, tf.float32)
        Qx = tf.cast(Qx, tf.float32)
        Sy = tf.cast(Sy, tf.int32)
        Qy = tf.cast(Qy, tf.int32)
        with tf.GradientTape() as tape:
            loss, acc = prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy)
        grads = tape.gradient(loss, encoder.trainable_variables)
        opt.apply_gradients(zip(grads, encoder.trainable_variables))
        return loss, acc

    for t in range(1, steps+1):
        Sx, Sy, Qx, Qy, _ = make_episode(X_tr, y_tr, N, K, Q, rng)
        loss, acc = _train_step(Sx, Sy, Qx, Qy)

        if t % val_every == 0:
            # episodic 验证（纯前向，不反传）
            accs = []
            for _ in range(val_episodes):
                Sx, Sy, Qx, Qy, _ = make_episode(X_val, y_val, N, K, Q, rng)
                _, a = prototypical_loss_and_acc(encoder, Sx, Sy, Qx, Qy)
                accs.append(float(a))
            val_acc = float(np.mean(accs))
            if val_acc > best_val:
                best_val = val_acc
                best_weights = encoder.get_weights()
            print(f"[step {t}] train_acc={float(acc):.3f}  val_acc={val_acc:.3f}")

    encoder.set_weights(best_weights)
    print(f"Loaded best encoder (val_acc={best_val:.3f})")
    return encoder


# =========================
# 5) Few-shot 推理与评测
# =========================
def predict_episode(encoder: Model, Sx, Sy, Qx, classes):
    """
    给定 episode 的支撑/查询（Sy 为 0..N-1），返回查询的原始类ID预测。
    """
    # 嵌入（已在模型中 L2 normalize）
    emb_s = encoder.predict(Sx, batch_size=256, verbose=0)
    emb_q = encoder.predict(Qx, batch_size=256, verbose=0)
    # 原型（按 Sy 的 0..N-1 顺序）
    N = len(np.unique(Sy))
    protos = np.stack([emb_s[Sy == c].mean(axis=0) for c in range(N)], axis=0)  # [N,D]
    D = cdist(emb_q, protos, metric="sqeuclidean")  # 稳定
    pred_idx = np.argmin(D, axis=1)                # 0..N-1
    return classes[pred_idx]                        # 映射回原始类ID


def episodic_eval(encoder: Model, X, y, episodes=200, N=5, K=5, Q=15, seed=0):
    rng = np.random.default_rng(seed)
    correct = 0; total = 0
    for _ in range(episodes):
        Sx, Sy, Qx, Qy, classes = make_episode(X, y, N, K, Q, rng)
        pred_labels = predict_episode(encoder, Sx, Sy, Qx, classes)
        # Qy 是 0..N-1 的内部标签，要映射为原始类ID比较
        true_labels = classes[Qy]
        correct += (pred_labels == true_labels).sum()
        total += len(true_labels)
    acc = correct / total
    return acc


# =========================
# 6)（可选）对角马氏距离推理（更稳）
# =========================
def predict_episode_mahalanobis_diag(encoder: Model, Sx, Sy, Qx, classes, eps=1e-3):
    emb_s = encoder.predict(Sx, batch_size=256, verbose=0)
    emb_q = encoder.predict(Qx, batch_size=256, verbose=0)
    N = len(np.unique(Sy))
    protos = []
    inv_vars = []  # 对角协方差的逆（方差 + eps 的倒数）
    for c in range(N):
        cls = emb_s[Sy == c]
        mu = cls.mean(axis=0)
        var = cls.var(axis=0)
        protos.append(mu)
        inv_vars.append(1.0 / (var + eps))
    protos = np.stack(protos)         # [N,D]
    inv_vars = np.stack(inv_vars)     # [N,D]
    # 逐类计算 (x - mu)^T diag(inv_vars) (x - mu)
    # -> d2[i,c] = sum_k (q[i,k]-mu[c,k])^2 * inv_vars[c,k]
    d2 = ((emb_q[:, None, :] - protos[None, :, :])**2 * inv_vars[None, :, :]).sum(axis=-1)  # [BQ,N]
    pred_idx = d2.argmin(axis=1)
    return classes[pred_idx]

def find_indices(array, max,min):
    indices = [i for i, x in enumerate(array) if x >=min and x<=max]
    return indices
class EMSA:
    """
    Extended Multiplicative Signal Augmentation
    Generates balanced batches of augmentated spectra
    """

    def __init__(self, std_of_params, wavenumbers, reference, order=2):
        """
        :param std_of_params: array of length (order+2), which
        :param reference: reference spectrum that was used in EMSC model
        :param order: order of emsc
        contains the std for each coefficient
        """
        self.order = order
        self.std_of_params = std_of_params
        self.ref = reference
        self.X = None
        self.A = None
        self.__create_x_and_a(wavenumbers)

    def generator(self, spectra, labels,
                  equalize_subsampling=False, shuffle=True,
                  batch_size=32):
        """ generates batches of transformed spectra"""
        spectra = np.asarray(spectra)
        labels = np.asarray(labels)

        if self.std_of_params is None:
            coefs = np.dot(self.A, spectra.T)
            self.std_of_params = coefs.std(axis=1)

        if equalize_subsampling:
            indexes = self.__rearrange_spectra(labels)
        else:
            indexes = np.arange(len(spectra))

        cur = 0
        while True:
            if shuffle:
                si = indexes[np.random.randint(len(indexes),
                                               size=batch_size)]
            else:
                si = indexes.take(range(cur, cur + batch_size),
                                  mode='wrap')
                cur += batch_size

            yield self.__batch_transform(spectra[si]), labels[si]

    def __rearrange_spectra(self, labels):
        """ returns indexes of data rearranged in the way of 'balance'"""
        classes = np.unique(labels, axis=0)

        if len(labels.shape) == 2:
            grouped = [np.where(np.all(labels == l, axis=1))[0]
                       for l in classes]
        else:
            grouped = [np.where(labels == l)[0] for l in classes]
        iters_cnt = max([len(g) for g in grouped])

        indexes = []
        for i in range(iters_cnt):
            for g in grouped:
                # take cyclic sample from group
                indexes.append(np.take(g, i, mode='wrap'))

        return np.array(indexes)

    def __create_x_and_a(self, wavenumbers):
        """
        Builds X matrix from spectra in such way that columns go as
        reference w^0 w^1 w^2 ... w^n, what corresponds to coefficients
        b, a, d, e, ...
        and caches the solution self.A = (X^T*X)^(-1)*X^T
        :param spectra:
        :param wavenumbers:
        :return: nothing, but creates two self.X and self.A
        """
        # squeeze wavenumbers to approx. range [-1; 1]
        # use if else to support uint types
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]
        half_rng = rng / 2
        normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

        self.polynomial_columns = [np.ones_like(wavenumbers)]
        for j in range(1, self.order + 1):
            self.polynomial_columns.append(normalized_wns ** j)

        self.X = np.stack((self.ref, *self.polynomial_columns), axis=1)
        self.A = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T)

    def __batch_transform(self, spectra):
        spectra_columns = spectra.T

        # b, a, d, e, ...

        coefs = np.dot(self.A, spectra_columns)
        residues = spectra_columns - np.dot(self.X, coefs)

        new_coefs = coefs.copy()

        # wiggle coefficients
        for i in range(len(coefs)):
            new_coefs[i] += np.random.normal(0,
                                             self.std_of_params[i],
                                             len(spectra))

        # Fix if multiplication parameter sampled negative
        mask = new_coefs[0] <= 0
        if np.any(mask):
            # resample multiplication parameter to be positive
            n_resamples = mask.sum()
            new_coefs[0][mask] = np.random.uniform(0, coefs[0][mask],
                                                   n_resamples)


        return (np.dot(self.X, new_coefs) + residues * new_coefs[0] / coefs[0]).T
def findSpectrum(spectrum,indexList,num):
    indices = [i for i, x in enumerate(indexList) if x==num]
    FindArray=spectrum[indices]
    return FindArray

def emsc(spectra: np.ndarray, wavenumbers: np.ndarray, order: int = 2,
         reference: np.ndarray = None,
         constituents: np.ndarray = None,
         return_coefs: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param order: order of polynomial
    :param reference: reference spectrum
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    print(spectra)
    print(reference)
    reference = reference[:, np.newaxis]

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

    polynomial_columns = [np.ones(len(wavenumbers))]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = np.stack(polynomial_columns).T

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = np.concatenate(columns, axis=1)
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    preprocessed_spectra = (reference + residues/coefs[0]).T

    if return_coefs:
        return preprocessed_spectra, coefs.T

    return preprocessed_spectra

def get_data():
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('dataset/D4_4_publication11.csv',
                                                                                      2, 1763)
    polymerName2, waveLength2, intensity2, polymerID2 = utils.parseDataForSecondDataset2(
        'dataset/new_SecondDataset2.csv')

    polymerName4, waveLength4, intensity4, polymerID4 = utils.parseData4th('dataset/FourthdatasetFollp-r3.csv')
    intensity3 = intensity2[0]

    #
    for i in range(len(intensity2)):
        intensity2[i] = intensity2[i][::-1]

    intensity3 = intensity2[0]
    max1 = max(waveLength)
    max2 = max(waveLength2)
    max3 = max(waveLength4)
    min1 = min(waveLength)
    min2 = min(waveLength2)
    min3 = min(waveLength4)
    maxwavelength = min(max1, max2, max3)
    minwavelenth = max(min1, min2, min3)
    print(maxwavelength, minwavelenth)
    indices0 = find_indices(waveLength, maxwavelength, minwavelenth)
    indices = find_indices(waveLength2, maxwavelength, minwavelenth)
    indices4 = find_indices(waveLength4, maxwavelength, minwavelenth)
    print('4', indices4)
    print('0', indices0)
    print('1.', indices)
    print(len(waveLength))
    chooseIdex01 = indices0[0]
    chooseIdex02 = indices0[-1]
    chooseIdex1 = indices[0]
    chooseIdex2 = indices[-1]
    chooseIdex41 = indices4[0]
    chooseIdex42 = indices4[-1]
    print('choose0', chooseIdex01, chooseIdex02)
    print('choose4', chooseIdex41, chooseIdex42)
    # chooseIdex1=len(waveLength2)-chooseIdex2
    # chooseIdex2=len(waveLength2)-chooseIdex1
    i = 0
    ppid = [i for i, item in enumerate(polymerID2) if item == 3]
    ppid1 = [i for i, item in enumerate(polymerName) if item == 'Poly(propylene)']
    ppid2 = [i for i, item in enumerate(polymerName4) if item == 'PP']
   # print('ppid2', ppid2)
    # ppid1=[i for i ,item in enumerate (polymerName) if item=='Poly(styrene)' ]

    ppidforadd = []
    for item in range(len(ppid)):
        ppidforadd.append(polymerID[ppid1][0])
    ppidforadd3 = []
    for item in range(len(ppid2)):
        ppidforadd3.append(polymerID[ppid1][0])
   # print('ppid3', ppidforadd3)
    #
    # for i in range(len(ppid)):
    #     print(ppid[i])
    intensityDataset1 = []
    for item in intensity:
        intensityDataset1.append(item[chooseIdex01:chooseIdex02])
    intensityDataset2 = []
    for item in intensity2:
        intensityDataset2.append(item[chooseIdex1:chooseIdex2])
    intensityDataset4 = []
    for item in intensity4:
        intensityDataset4.append(item[chooseIdex41:chooseIdex42])
    #print('max wavelength', waveLength2[chooseIdex2])
    intensityDataset2 = np.array(intensityDataset2)
    intensityDataset4=intensityDataset4[::-1]
    for i in range(len(intensityDataset2)):
        intensityDataset2[i]=intensityDataset2[i][::-1]
    # print('intensityDataset2',intensityDataset2.shape)
    waveLength = np.array(waveLength, dtype=np.float)
    # print('wavelength',waveLength)
    waveLength2 = np.array(waveLength2, dtype=np.float)

    waveLength4 = np.array(waveLength4, dtype=np.float)
    # print('wavelength2',waveLength2)
    waveLength = waveLength[chooseIdex01:chooseIdex02]
   # print('wavelength', waveLength)
    waveLength3 = waveLength2[chooseIdex1:chooseIdex2]
    #waveLength3 = waveLength3[::-1]
    # print('wavelength3', waveLength3)
    waveLength4 = waveLength4[::-1][chooseIdex41:chooseIdex42]
    waveLength4 = waveLength4[::-1]
    waveLength42 = waveLength4[chooseIdex41:chooseIdex42]

    # print('wavelength4', waveLength4)
    # print('wavelength42', waveLength42)
    # print(waveLength.shape)
    # x = np.linspace(0, 10, 11)
    # # x=[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
    # y = np.sin(x)
    # xnew2=np.linspace(max(waveLength2),min(waveLength2),1000)

    # xnew=np.linspace(0,10,8)

    # pl.plot(waveLength2,intensity3,"ro")
    # xnew2=np.linspace(max(waveLength2),min(waveLength2),1000)
    # #for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
    # for kind in [ "cubic"]:  # 插值方式
    #     #"nearest","zero"为阶梯插值
    #     #slinear 线性插值
    #     #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    #     f=interpolate.interp1d(waveLength2,intensity3,kind=kind)
    #     # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    #     ynew=f(xnew2)
    #     print(ynew)
    #
    #     pl.plot(xnew2,ynew,label=str(kind))
    maxwavelength = min(max(waveLength3), max(waveLength4), max(waveLength))
    minwavelenth=max(min(waveLength3),min(waveLength4),min(waveLength))
    xnew3 = np.linspace(max(waveLength3), min(waveLength3), 2000)
    xnew4 = np.linspace(max(waveLength4), min(waveLength4), 2000)
    xnew2 = np.linspace(max(waveLength), min(waveLength), 2000)
    xnewFinal = np.linspace(maxwavelength, minwavelenth, 2000)
    # print('xnew2',xnew2)
    #
    # print('xnew4', xnew4)
    # print('xnew3', xnew3)
    #
    # print('xnewFinal', xnewFinal)
    # for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
    for kind in ["cubic"]:  # 插值方式
        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        f2 = interpolate.interp1d(waveLength3, intensityDataset2, kind=kind)
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        f = interpolate.interp1d(waveLength, intensityDataset1, kind=kind)
        f4 = interpolate.interp1d(waveLength4, intensityDataset4, kind=kind)

        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

        ynew = f(xnewFinal)
        # for item in ynew:
        #     pl.plot(xnew2, item)
        ynew2 = f2(xnewFinal)
        ynew3 = f4(xnewFinal)
        # print('ynew2shape', ynew2.shape)
        # print('ynew3shape', ynew3.shape)
    #     for item in ynew2:
    #
    #         pl.plot(xnew3,item)
    #
    # pl.legend(loc="lower right")
    # pl.show()
    PPfirstdataset = findSpectrum(ynew, polymerID, 3)
    PPseconddataset = findSpectrum(ynew2, polymerID2, 3)
    PPthriddataset = findSpectrum(ynew3, polymerID4, 9)
    PPfirstdataset = np.array(PPfirstdataset)
    PPseconddataset = np.array(PPseconddataset)
    PPthriddataset = np.array(PPthriddataset)
    # for i in range(len(PPseconddataset)):
    #     PPseconddataset[i] = np.flip(PPseconddataset[i])
    # print('PPfirst', PPfirstdataset.shape)
    # print('PPsecond', PPseconddataset.shape)
    # for i in range(len(PPfirstdataset)):
    #     plt.plot(xnew2, PPfirstdataset[i], 'r')
    # for i in range(len(PPseconddataset)):
    #     plt.plot(xnew3, PPseconddataset[i], 'y')
    # for i in range(len(PPthriddataset)):
    #     plt.plot(xnew4, PPthriddataset[i], 'b')
    # plt.ylim(0,1.1)
    # plt.gca().invert_xaxis()
    # plt.show()
    return ynew,ynew2,ynew3,polymerID,polymerID2,polymerID4,polymerName,polymerName2,polymerName4,xnewFinal

firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()
w4,  X4,  y4,  p4 = readFromPlastics500('dataset/FTIR_PLastics500_c4.csv')
w5,  X5,  y5,  p5 = readFromPlastics500('dataset/FTIR_PLastics500_c8.csv')
def main():
    # === 1) 选一个数据集（例如 c8）===
    X, y = X5, y5
    # 归一化（与训练/测试保持一致；若你前面已有 EMSC/EMSA，则保持相同策略）
    # 例如：每条谱做最大值归一化
    X = X / (np.max(X, axis=1, keepdims=True) + 1e-8)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    encoder = make_encoder(input_len=X.shape[1], emb_dim=128)

    # === 2) episodic 训练 ===
    encoder = train_protonet(
        encoder, X_tr, y_tr, X_val, y_val,
        steps=2000, N=5, K=5, Q=15, lr=1e-3, seed=0,
        val_every=100, val_episodes=50
    )

    # === 3) episodic 评测（同域）===
    acc = episodic_eval(encoder, X_val, y_val, episodes=200, N=5, K=5, Q=15, seed=1)
    print(f"[c8 同域 episodic acc] {acc:.4f}")

    # === 4) 跨域 few-shot 评测（例如在 c4）===
    X_t, y_t = X4, y4
    X_t = X_t / (np.max(X_t, axis=1, keepdims=True) + 1e-8)

    acc_cross = episodic_eval(encoder, X_t, y_t, episodes=200, N=5, K=5, Q=15, seed=2)
    print(f"[c8 训练 -> c4 评测 episodic acc] {acc_cross:.4f}")

    # === 5) 单个 episode 推理（拿到预测标签）===
    Sx, Sy, Qx, Qy, classes = make_episode(X_t, y_t, N=6, K=5, Q=10)
    pred_euclid = predict_episode(encoder, Sx, Sy, Qx, classes)
    pred_maha   = predict_episode_mahalanobis_diag(encoder, Sx, Sy, Qx, classes)
    true_labels = classes[Qy]
    print("Euclid acc (one episode):", (pred_euclid == true_labels).mean())
    print("Diag-Mahalanobis acc (one episode):", (pred_maha == true_labels).mean())



if __name__ == "__main__":
    # TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)
    main()
