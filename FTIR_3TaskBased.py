from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, \
    GlobalAveragePooling1D
from tensorflow.keras.models import Model
import pandas as pd
from scipy import interpolate
from utils import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Union as U, Tuple as T
from FTIR_ReaddataFrom500C4 import readFromPlastics500
from keras import Sequential
from keras.layers import Reshape
from keras.models import load_model
from sklearn.metrics import confusion_matrix
def find_indices(array, max,min):
    indices = [i for i, x in enumerate(array) if x >=min and x<=max]
    return indices
def findSpectrum(spectrum,indexList,num):
    indices = [i for i, x in enumerate(indexList) if x==num]
    FindArray=spectrum[indices]
    return FindArray
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
def euclidean_distance(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1))
def classify(query_embeddings, prototypes):
    distances = euclidean_distance(np.expand_dims(query_embeddings, 1), np.expand_dims(prototypes, 0))
    return np.argmin(distances, axis=1)
def dataAugmenation3(intensity, polymerID, waveLength, pName, randomSeed):
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=randomSeed)
    waveLength = np.array(waveLength, dtype=np.float)
    datas = []
    datas2 = []
    PN = []
    for item in pName:
        if item not in PN:
            PN.append(item)
    polymerMID = []
    for item in polymerID:
        if item not in polymerMID:
            polymerMID.append(item)
    indicesList=[]
    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(y_train) if id == polymerMID[n]]
        indicesList.append(indicesPS)
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    for itr in range(len(PN)):
        _, coefs_ = emsc(
            datas[itr], waveLength, reference=None,
            order=2,
            return_coefs=True)

        coefs_std = coefs_.std(axis=0)
        # print(polymerMID[itr])

        label = indicesList[itr]
        reference = datas[itr].mean(axis=0)
        emsa = EMSA(coefs_std, waveLength, reference, order=2)

        generator = emsa.generator(datas[itr], label,
                                   equalize_subsampling=False, shuffle=False,
                                   batch_size=200)

        augmentedSpectrum = []
        for i, batch in enumerate(generator):
            if i > 2:
                break
            augmented = []
            for augmented_spectrum, label in zip(*batch):
                plt.plot(waveLength, augmented_spectrum, label=label)
                augmented.append(augmented_spectrum)
            augmentedSpectrum.append(augmented)
            # plt.gca().invert_xaxis()
            # plt.legend()
            # plt.show()
        augmentedSpectrum = np.array(augmentedSpectrum)
        y_add = []
        for item in augmentedSpectrum[0]:
            y_add.append(polymerMID[itr])
        from sklearn.preprocessing import normalize
        augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
        x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
        y_train = np.concatenate((y_train, y_add), axis=0)
    return x_train, y_train, x_test, y_test

# def train_proto_net2(model, support_data, support_labels, query_data, query_labels, num_classes, epochs=10):
#     optimizer = tf.keras.optimizers.Adam()
#
#     for epoch in range(epochs):
#         with tf.GradientTape() as tape:
#             tape.watch(model.trainable_variables)
#             support_embeddings = model(support_data, training=True)
#             query_embeddings = model(query_data, training=True)
#             loss = proto_loss(support_embeddings, query_embeddings, support_labels, query_labels, num_classes)
#
#         gradients = tape.gradient(loss, model.trainable_variables)
#         if any(g is None for g in gradients):
#             print("存在一个或多个梯度为 None，检查模型和损失函数的实现。")
#             for var, grad in zip(model.trainable_variables, gradients):
#                 print(f"{var.name}, gradient: {grad}")
#         else:
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#         print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}')
#

def proto_loss1(support_embeddings, query_embeddings, support_labels, query_labels):
    num_classes = tf.reduce_max(query_labels) + 1
    prototypes = []
    for c in range(num_classes):
        class_mask = tf.equal(support_labels, c)
        class_embeddings = tf.boolean_mask(support_embeddings, class_mask)
        prototype = tf.reduce_mean(class_embeddings, axis=0)
        prototypes.append(prototype)
    prototypes = tf.stack(prototypes)

    distances = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=2)
    log_p_y = tf.nn.log_softmax(-distances)

    return -tf.reduce_mean(tf.reduce_sum(log_p_y * tf.one_hot(query_labels, depth=num_classes), axis=1))
# def proto_loss(support_embeddings, query_embeddings, support_labels, query_labels):
#     num_classes = tf.reduce_max(query_labels) + 1
#     prototypes = []
#     for c in range(num_classes):
#         class_mask = tf.equal(support_labels, c)
#         class_embeddings = tf.boolean_mask(support_embeddings, class_mask)
#         prototype = tf.reduce_mean(class_embeddings, axis=0)
#         prototypes.append(prototype)
#     prototypes = tf.stack(prototypes)
#
#     distances = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=2)
#     log_p_y = tf.nn.log_softmax(-distances)






from tensorflow.keras.callbacks import ReduceLROnPlateau


from tensorflow.keras.optimizers.schedules import ExponentialDecay


# 检查支持集和查询集数据
def dataAugmenation(intensity,polymerID,waveLength,pName,randomSeed):
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=randomSeed)
    waveLength = np.array(waveLength, dtype=np.float)
    datas = []
    datas2 = []
    PN = []
    for item in pName:
        if item not in PN:
            PN.append(item)
    polymerMID=[]
    for item in polymerMID:
        if item not in PN:
            polymerMID.append(item)

    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(y_train) if id == n]
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    for itr in range(0, len(PN)):
        _, coefs_ = emsc(
            datas[itr], waveLength, reference=None,
            order=2,
            return_coefs=True)

        coefs_std = coefs_.std(axis=0)
        indicesPS = [l for l, id in enumerate(y_train) if id == itr]
        label = y_train[indicesPS]
        reference = datas[itr].mean(axis=0)
        emsa = EMSA(coefs_std, waveLength, reference, order=2)

        generator = emsa.generator(datas[itr], label,
                                   equalize_subsampling=False, shuffle=False,
                                   batch_size=200)

        augmentedSpectrum = []
        for i, batch in enumerate(generator):
            if i > 2:
                break
            augmented = []
            for augmented_spectrum, label in zip(*batch):
                plt.plot(waveLength, augmented_spectrum, label=label)
                augmented.append(augmented_spectrum)
            augmentedSpectrum.append(augmented)
            # plt.gca().invert_xaxis()
            # plt.legend()
            # plt.show()
        augmentedSpectrum = np.array(augmentedSpectrum)
        y_add = []
        for item in augmentedSpectrum[0]:
            y_add.append(itr)
        from sklearn.preprocessing import normalize
        augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
        x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
        y_train = np.concatenate((y_train, y_add), axis=0)
    return x_train,y_train,x_test,y_test


def dataAugmenation2(intensity, polymerID, waveLength, pName, randomSeed):
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=randomSeed)
    waveLength = np.array(waveLength, dtype=np.float)
    datas = []
    datas2 = []
    PN = []
    for item in pName:
        if item not in PN:
            PN.append(item)
    polymerMID = []
    for item in polymerID:
        if item not in polymerMID:
            polymerMID.append(item)
    indicesList=[]
    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(y_train) if id == polymerMID[n]]
        indicesList.append(indicesPS)
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    for itr in range(len(PN)):
        _, coefs_ = emsc(
            datas[itr], waveLength, reference=None,
            order=2,
            return_coefs=True)

        coefs_std = coefs_.std(axis=0)
        # print(polymerMID[itr])

        label = indicesList[itr]
        reference = datas[itr].mean(axis=0)
        emsa = EMSA(coefs_std, waveLength, reference, order=2)

        generator = emsa.generator(datas[itr], label,
                                   equalize_subsampling=False, shuffle=False,
                                   batch_size=300)

        augmentedSpectrum = []
        for i, batch in enumerate(generator):
            if i > 2:
                break
            augmented = []
            for augmented_spectrum, label in zip(*batch):
                plt.plot(waveLength, augmented_spectrum, label=label)
                augmented.append(augmented_spectrum)
            augmentedSpectrum.append(augmented)
            # plt.gca().invert_xaxis()
            # plt.legend()
            # plt.show()
        augmentedSpectrum = np.array(augmentedSpectrum)
        y_add = []
        for item in augmentedSpectrum[0]:
            y_add.append(polymerMID[itr])
        from sklearn.preprocessing import normalize
        augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
        x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
        y_train = np.concatenate((y_train, y_add), axis=0)
    return x_train, y_train, x_test, y_test
def dataAugmenation3(intensity, polymerID, waveLength, pName, randomSeed):
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=randomSeed)
    waveLength = np.array(waveLength, dtype=np.float)
    datas = []
    datas2 = []
    PN = []
    for item in pName:
        if item not in PN:
            PN.append(item)
    polymerMID = []
    for item in polymerID:
        if item not in polymerMID:
            polymerMID.append(item)
    indicesList=[]
    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(y_train) if id == polymerMID[n]]
        indicesList.append(indicesPS)
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    for itr in range(len(PN)):
        _, coefs_ = emsc(
            datas[itr], waveLength, reference=None,
            order=2,
            return_coefs=True)

        coefs_std = coefs_.std(axis=0)
        # print(polymerMID[itr])

        label = indicesList[itr]
        reference = datas[itr].mean(axis=0)
        emsa = EMSA(coefs_std, waveLength, reference, order=2)

        generator = emsa.generator(datas[itr], label,
                                   equalize_subsampling=False, shuffle=False,
                                   batch_size=200)

        augmentedSpectrum = []
        for i, batch in enumerate(generator):
            if i > 2:
                break
            augmented = []
            for augmented_spectrum, label in zip(*batch):
                plt.plot(waveLength, augmented_spectrum, label=label)
                augmented.append(augmented_spectrum)
            augmentedSpectrum.append(augmented)
            # plt.gca().invert_xaxis()
            # plt.legend()
            # plt.show()
        augmentedSpectrum = np.array(augmentedSpectrum)
        y_add = []
        for item in augmentedSpectrum[0]:
            y_add.append(polymerMID[itr])
        from sklearn.preprocessing import normalize
        augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
        x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
        y_train = np.concatenate((y_train, y_add), axis=0)
    return x_train, y_train, x_test, y_test


def print_data_statistics(data_list, labels_list):
    for i, (data, labels) in enumerate(zip(data_list, labels_list)):
        print(f"Task {i + 1} Data Statistics:")
        print(f"  Data shape: {data.shape}, Labels shape: {labels.shape}")
        print(f"  Data mean: {np.mean(data)}, Data std: {np.std(data)}")
        print(f"  Labels mean: {np.mean(labels)}, Labels std: {np.std(labels)}")
        print(f"  Labels unique: {np.unique(labels)}")
from keras.layers import MaxPool1D
# def build_feature_extractor(input_shape):
#
#     model = Sequential([
#     Reshape((input_shape, 1), input_shape=(input_shape,)),
#     Conv1D(32, 32, activation='relu', input_shape=(input_shape, 1), padding="same"),
#     MaxPool1D(pool_size=3, strides=3),
#     # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
#     # MaxPool1D(pool_size=3, strides=3),
#     # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
#     # MaxPool1D(pool_size=3, strides=3),
#     Flatten(),
#     # Dense(128, activation='relu'),
#     # Dense(128, activation='relu'),
#     Dense(128, activation='softmax')])
#     return model
def build_complex_feature_extractor(input_shape):
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
def build_feature_extractor(input_shape):
    initializer = tf.keras.initializers.GlorotUniform()
    inputs = Input(shape=(input_shape,))
    x = Dense(512, activation='relu', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    features = Dense(64, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    return Model(inputs, features)


def get_feature_extractor(model):
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-1].input)
    return feature_extractor


# def batch_generator(data, labels, batch_size):
#     unique_labels = np.unique(labels)
#     label_to_data = {label: data[labels == label] for label in unique_labels}
#     label_to_indices = {label: np.arange(len(label_to_data[label])) for label in unique_labels}
#
#     for label in label_to_indices:
#         np.random.shuffle(label_to_indices[label])
#
#     max_batches = min(len(indices) // (batch_size // len(unique_labels)) for indices in label_to_indices.values())
#
#     for batch_idx in range(max_batches):
#         batch_data = []
#         batch_labels = []
#         for label in unique_labels:
#             start_idx = batch_idx * (batch_size // len(unique_labels))
#             end_idx = (batch_idx + 1) * (batch_size // len(unique_labels))
#             batch_data.append(label_to_data[label][label_to_indices[label][start_idx:end_idx]])
#             batch_labels.append(np.full((batch_size // len(unique_labels)), label))
#         yield np.vstack(batch_data), np.concatenate(batch_labels)

# def train_feature_extractor(feature_extractor, support_data_list, support_labels_list, num_classes_per_task, epochs=10, batch_size=32):
#     initial_learning_rate = 0.001
#     lr_schedule = ExponentialDecay(
#         initial_learning_rate,
#         decay_steps=1000,
#         decay_rate=0.96,
#         staircase=True)
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#     min_lr = 1e-6
#     factor = 0.5
#     patience = 10
#     wait = 0
#     best_loss = np.inf
#
#     classifier_layers = [Dense(num_classes, activation='softmax') for num_classes in num_classes_per_task]
#
#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs} start")
#         total_loss = 0.0
#         valid_batches = 0
#
#         for task in range(len(support_data_list)):
#             support_data = np.array(support_data_list[task], dtype=np.float32)
#             support_labels = np.array(support_labels_list[task], dtype=np.int32)
#
#             for batch_data, batch_labels in batch_generator(support_data, support_labels, batch_size):
#                 with tf.GradientTape() as tape:
#                     embeddings = feature_extractor(batch_data, training=True)
#                     task_outputs = classifier_layers[task](embeddings)
#                     classification_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, task_outputs)
#                     loss = tf.reduce_mean(classification_loss)
#
#                     if tf.math.is_nan(loss):
#                         print(f"NaN detected in loss for Task {task + 1} at Epoch {epoch + 1}, skipping this task")
#                         continue
#
#                     total_loss += loss
#                     valid_batches += 1
#
#                 gradients = tape.gradient(loss, feature_extractor.trainable_variables + classifier_layers[task].trainable_variables)
#                 print(f"Epoch {epoch + 1}/{epochs}, Task {task + 1}, Batch Loss: {loss.numpy()}")
#                 for g in gradients:
#                     if g is not None:
#                         print(f"Gradient max value: {tf.reduce_max(g).numpy()}, min value: {tf.reduce_min(g).numpy()}")
#                         if tf.reduce_max(tf.abs(g)) > 1e6:
#                             print("Gradient explosion detected!")
#                             return
#                 gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients if g is not None]
#                 optimizer.apply_gradients(zip(gradients, feature_extractor.trainable_variables + classifier_layers[task].trainable_variables))
#
#         if valid_batches > 0:
#             epoch_loss = total_loss / valid_batches
#             print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss.numpy()}')
#         else:
#             epoch_loss = float("inf")
#             print(f'Epoch {epoch + 1}/{epochs}, No valid batches, Average Loss: {epoch_loss}')
#
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             wait = 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 current_lr = optimizer.learning_rate.numpy()
#                 new_lr = max(current_lr * factor, min_lr)
#                 optimizer.learning_rate.assign(new_lr)
#                 print(f"Learning rate reduced to {new_lr}")
#                 wait = 0
#         print(f"Epoch {epoch + 1}/{epochs} end")

def batch_generator(data, labels, batch_size):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))  # 确保结束索引不会超出范围
        excerpt = indices[start_idx:end_idx]
        yield data[excerpt], labels[excerpt]
def proto_loss(support_embeddings, support_labels):
    prototypes, unique_labels = compute_prototypes6(support_embeddings, support_labels)
    distances = np.linalg.norm(np.expand_dims(support_embeddings, axis=1) - prototypes, axis=2)
    log_p_y = tf.nn.log_softmax(-tf.convert_to_tensor(distances, dtype=tf.float32), axis=-1)

    indices = tf.convert_to_tensor([np.where(unique_labels == label)[0][0] for label in support_labels], dtype=tf.int32)
    loss = -tf.reduce_mean(tf.gather(log_p_y, indices, axis=1))
    return loss
from tensorflow.keras.backend import get_value, set_value
def train_feature_extractor_with_proto_loss(feature_extractor, support_data_list, support_labels_list, epochs=10, batch_size=32):
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    min_lr = 1e-6
    factor = 0.5
    patience = 10
    wait = 0
    best_loss = np.inf

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} start")
        total_loss = 0.0
        valid_batches = 0

        for task in range(len(support_data_list)):
            support_data = np.array(support_data_list[task], dtype=np.float32)
            support_labels = np.array(support_labels_list[task], dtype=np.int32)

            for batch_data, batch_labels in batch_generator(support_data, support_labels, batch_size):
                with tf.GradientTape() as tape:
                    support_embeddings = feature_extractor(batch_data, training=True)
                    loss = proto_loss(support_embeddings, batch_labels)

                    if tf.math.is_nan(loss):
                        print(f"NaN detected in loss for Task {task + 1} at Epoch {epoch + 1}, skipping this task")
                        continue

                    total_loss += loss
                    valid_batches += 1

                gradients = tape.gradient(loss, feature_extractor.trainable_variables)
                print(f"Epoch {epoch + 1}/{epochs}, Task {task + 1}, Batch Loss: {loss.numpy()}")
                for g in gradients:
                    if g is not None:
                        print(f"Gradient max value: {tf.reduce_max(g).numpy()}, min value: {tf.reduce_min(g).numpy()}")
                        if tf.reduce_max(tf.abs(g)) > 1e6:
                            print("Gradient explosion detected!")
                            return
                gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients if g is not None]
                optimizer.apply_gradients(zip(gradients, feature_extractor.trainable_variables))

        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss.numpy()}')
        else:
            epoch_loss = float("inf")
            print(f'Epoch {epoch + 1}/{epochs}, No valid batches, Average Loss: {epoch_loss}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                current_lr = optimizer.learning_rate.numpy()
                new_lr = max(current_lr * factor, min_lr)
                optimizer.learning_rate.assign(new_lr)
                print(f"Learning rate reduced to {new_lr}")
                wait = 0
        print(f"Epoch {epoch + 1}/{epochs} end")
def check_data(data_list):
    for data in data_list:
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Data contains NaN or Inf values!")
            return False
    return True
def compute_prototypes6(embeddings, labels):
    embeddings_np = embeddings if isinstance(embeddings, np.ndarray) else embeddings.numpy()
    labels_np = labels if isinstance(labels, np.ndarray) else labels.numpy()
    unique_labels = np.unique(labels_np)
    prototypes = []
    for label in unique_labels:
        class_embeddings = embeddings_np[labels_np == label]
        class_prototype = np.mean(class_embeddings, axis=0)
        prototypes.append(class_prototype)
    prototypes = np.stack(prototypes)
    return prototypes, unique_labels
def classify_with_feature_extractor(feature_extractor, support_data_list, support_labels_list, query_data_list):
    predictions = []
    for task in range(len(support_data_list)):
        support_data = np.array(support_data_list[task], dtype=np.float32)
        query_data = np.array(query_data_list[task], dtype=np.float32)
        support_labels = np.array(support_labels_list[task], dtype=np.int32)

        support_embeddings = feature_extractor(support_data, training=False).numpy()
        query_embeddings = feature_extractor(query_data, training=False).numpy()

        prototypes, unique_labels = compute_prototypes6(support_embeddings, support_labels)
        distances = np.linalg.norm(np.expand_dims(query_embeddings, axis=1) - prototypes, axis=2)
        preds = np.argmin(distances, axis=1)
        predicted_labels = unique_labels[preds]
        predictions.append(predicted_labels)
    return predictions

if __name__ == '__main__':

    firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()

    x_train1,y_train1,x_test1,y_test1=dataAugmenation(firstData,pid1,wavenumber,pname1,1)

    x_train3, y_train3, x_test3, y_test3 = dataAugmenation2(thirdData, pid3, wavenumber, pname3, 1)
    # # fileName='FTIR_PLastics500_c4.csv'
    wavenumber4, forthData, pid4, pname4 = readFromPlastics500('FTIR_PLastics500_c4.csv')
    wavenumber5, fifthData, pid5, pname5 = readFromPlastics500('FTIR_PLastics500_c8.csv')
    #x_train3, x_test3, y_train3, y_test3 = train_test_split(thirdData, pid3, test_size=0.3, random_state=1)
    # #x_train1, x_test1, y_train1, y_test1 = train_test_split(firstData, pid1, test_size=0.3, random_state=1)
    x_train2, y_train2, x_test2, y_test2 = dataAugmenation2(secondData, pid2, wavenumber, pname2, 1)
    # x_train4, y_train4, x_test4, y_test4 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
    #x_train2, x_test2, y_train2, y_test2 = train_test_split(secondData, pid2, test_size=0.3, random_state=1)
    # print(forthData.shape)
    # x_train1, y_train1, x_test1, y_test1 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
    x_train4, x_test4, y_train4, y_test4 = train_test_split(forthData, pid4, test_size=0.7, random_state=1)
    x_train5, x_test5, y_train5, y_test5 = train_test_split(fifthData, pid5, test_size=0.7, random_state=1)

    # fileName = '4th_Dataset'
    # print('num_class', np.unique(pid4))
    # num_classes = len(np.unique(pid1))
    # # model = build_conv_extractor(len(x_train1[0]))
    # model = build_feature_extractor5(len(x_train1[0]), 32)
    #
    # model.summary()
    #
    # train_proto_net4(model, x_train1, y_train1, x_test1, y_test1, epochs=500)
    # support_embeddings = model.predict(x_train1)
    # query_embeddings = model.predict(x_test1)
    # prototypes = compute_prototypes(support_embeddings, y_train1, num_classes)
    # predicted_labels = classify(query_embeddings, prototypes)
    # extrModelName = 'model/build_feature_extractor' + fileName + '_model3.h5'
    # model.save(extrModelName)
    # # Evaluate accuracy
    # accuracy = np.mean(predicted_labels == y_test1)
    # print(f'Accuracy: {accuracy}')
    #
    # cm = confusion_matrix(y_test1, predicted_labels)
    # print(cm)
    # PN2 = []
    # for item in pname1:
    #     if item not in PN2:
    #         PN2.append(item)
    # print(PN2)
    # utils.plot_confusion_matrix(cm, PN2, '1st dataset')
    # # 示例数据
    # input_shape = 2000
    # num_tasks = 3
    # num_classes_per_task = [11, 4, 14]
    #
    # support_data_list=[x_train1,x_train2,x_train3]
    #
    # support_labels_list=[y_train1,y_train2,y_train3]
    # query_data_list = [x_test1, x_test2, x_test3]
    # #
    # query_labels_list = [y_test1, y_test2, y_test3]
    input_shape = 2000
    num_tasks = 3
    num_classes_per_task = [11, 4]

    support_data_list = [x_train1, x_train2]

    support_labels_list = [y_train1, y_train2]
    query_data_list = [x_test1, x_test2]
    #
    query_labels_list = [y_test1, y_test2]

    if not check_data(support_data_list) or not check_data(query_data_list):
        raise ValueError("dadad"
                         "dada"
                         "dada"
                         "dad"
                         "dada"
                         "dada"
                         "Data contains NaN or Inf values")

    # # num_support_samples = 5
    # # num_query_samples = 15
    # #
    # # support_data_list = [np.random.rand(num_classes_per_task[i] * num_support_samples, input_shape) for i in
    # #                      range(num_tasks)]
    # # support_labels_list = [np.array([[j] * num_support_samples for j in range(num_classes_per_task[i])]).flatten() for i in
    # #                        range(num_tasks)]
    # # query_data_list = [np.random.rand(num_classes_per_task[i] * num_query_samples, input_shape) for i in range(num_tasks)]
    # # query_labels_list = [np.array([[j] * num_query_samples for j in range(num_classes_per_task[i])]).flatten() for i in
    # #                      range(num_tasks)]
    # #
    # 构建多任务模型
    # model = build_multi_task_model(input_shape, num_tasks, num_classes_per_task)
    feature_extractor = build_feature_extractor(input_shape)
    # model, feature_extractor = build_multi_task_model6(input_shape, num_classes_per_task)
    #
    # # 训练特征提取器
    # #train_multi_task_model6(model, feature_extractor, support_data_list, support_labels_list, query_data_list,
    # #                         query_labels_list, num_classes_per_task, epochs=1000)
    #
    # # 训练多任务模型
    # # train_multi_task_model2(model, support_data_list, support_labels_list, query_data_list, query_labels_list,
    # #                        num_classes_per_task, epochs=1000)
    # #model.save('model/multi_task_model_feature_extractor_model1.h5')
    #
    #
    # # 初始化回调
    #
    # # 使用自定义训练过程
    # # custom_training_loop(model, support_data_list, support_labels_list, query_data_list,
    # #                      query_labels_list, num_classes_per_task, epochs=200,batch_size=32)
    # train_feature_extractor(feature_extractor, support_data_list, support_labels_list, num_classes_per_task, epochs=1,
    #                        batch_size=32)
    # train_feature_extractor(feature_extractor, support_data_list, support_labels_list, num_classes_per_task, epochs=100,
    #                         batch_size=32)

    train_feature_extractor_with_proto_loss(feature_extractor,support_data_list,support_labels_list, epochs=100, batch_size=8)
    # # custom_training_loop(model, support_data_list, support_labels_list, query_data_list, query_labels_list,
    # #                      num_classes_per_task, epochs=200, batch_size=32)
    # feature_extractor=get_feature_extractor(model)
    feature_extractor.save('model/multi_task_model_feature_extractor_model5.h5')
    # 评估多任务模型的准确性
    feature_extractor=load_model('model/multi_task_model_feature_extractor_model5.h5')
    predictions = classify_with_feature_extractor(feature_extractor, support_data_list, support_labels_list, query_data_list)
    # for i, predicted_labels in enumerate(predictions):
    #     accuracy = np.mean(predicted_labels.numpy() == query_labels_list[i])
    #     print(f'Task {i} Accuracy: {accuracy}')

    for i, predicted_labels in enumerate(predictions):
        accuracy = np.mean(predicted_labels == query_labels_list[i])
        print(f'Task {i} Accuracy: {accuracy}')
    # model=load_model('model/multi_task_model_build_simple_feature_extractor_model2.h5')
    support_data_list = [x_train5]

    support_labels_list = [y_train5]
    query_data_list = [x_test5]

    query_labels_list = [y_test5]

    predictions = classify_with_feature_extractor(feature_extractor, support_data_list, support_labels_list, query_data_list)
    # for i, predicted_labels in enumerate(predictions):
    #     accuracy = np.mean(predicted_labels.numpy() == query_labels_list[i])
    #     print(f'Task {i} Accuracy: {accuracy}')
    for i, predicted_labels in enumerate(predictions):
        accuracy = np.mean(predicted_labels == query_labels_list[i])
        print(f'Task {i} Accuracy: {accuracy}')
    cm = confusion_matrix(y_test5, predicted_labels)
    PN2 = []
    for item in pname5:
        if item not in PN2:
            PN2.append(item)
    print(PN2)
    utils.plot_confusion_matrix(cm, PN2, '5th dataset')

    support_data_list = [x_train3]

    support_labels_list = [y_train3]
    query_data_list = [x_test3]

    query_labels_list = [y_test3]

    predictions = classify_with_feature_extractor(feature_extractor, support_data_list, support_labels_list, query_data_list)
    # for i, predicted_labels in enumerate(predictions):
    #     accuracy = np.mean(predicted_labels.numpy() == query_labels_list[i])
    #     print(f'Task {i} Accuracy: {accuracy}')
    for i, predicted_labels in enumerate(predictions):
        accuracy = np.mean(predicted_labels == query_labels_list[i])
        print(f'Task {i} Accuracy: {accuracy}')
    cm = confusion_matrix(y_test3, predicted_labels)
    PN2 = []
    for item in pname3:
        if item not in PN2:
            PN2.append(item)
    print(PN2)
    utils.plot_confusion_matrix(cm, PN2, '1st dataset')

    support_data_list = [x_train4]

    support_labels_list = [y_train4]
    query_data_list = [x_test4]

    query_labels_list = [y_test4]

    predictions = classify_with_feature_extractor(feature_extractor, support_data_list, support_labels_list,
                                                  query_data_list)
    # for i, predicted_labels in enumerate(predictions):
    #     accuracy = np.mean(predicted_labels.numpy() == query_labels_list[i])
    #     print(f'Task {i} Accuracy: {accuracy}')
    for i, predicted_labels in enumerate(predictions):
        accuracy = np.mean(predicted_labels == query_labels_list[i])
        print(f'Task {i} Accuracy: {accuracy}')
    cm = confusion_matrix(y_test4, predicted_labels)
    PN2 = []
    for item in pname4:
        if item not in PN2:
            PN2.append(item)
    print(PN2)
    utils.plot_confusion_matrix(cm, PN2, '4th dataset')