import numpy
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
def find_indices(array, max,min):
    indices = [i for i, x in enumerate(array) if x >=min and x<=max]
    return indices
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    #     for item in ynew2:
    #
    #         pl.plot(xnew3,item)
    #
    # pl.legend(loc="lower right")
    # pl.show()
# 定义保存和加载类原型的辅助函数
def save_prototypes(prototypes, filename):
    with open(filename, 'wb') as file:
        pickle.dump(prototypes, file)


def load_prototypes(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# 定义生成支撑集和查询集的辅助函数
def generate_support_and_query_sets(data, labels, n_way, k_shot, n_query):
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

    classes = np.unique(labels)
    print('classes unique', classes)

    selected_classes = np.random.choice(classes, n_way, replace=False)

    for class_id in selected_classes:
        class_data = data[labels == class_id]
        indices = np.random.permutation(class_data.shape[0])

        support_samples = class_data[indices[:k_shot]]
        query_samples = class_data[indices[k_shot:k_shot + n_query]]

        support_set.append(support_samples)
        query_set.append(query_samples)
        support_labels.append([class_id] * k_shot)
        query_labels.append([class_id] * n_query)

    support_set = np.concatenate(support_set, axis=0)
    query_set = np.concatenate(query_set, axis=0)
    support_labels = np.concatenate(support_labels, axis=0)
    query_labels = np.concatenate(query_labels, axis=0)

    return support_set, support_labels, query_set, query_labels


# 定义计算类原型的辅助函数
def compute_prototypes(support_set, support_labels, model):
    support_embeddings = model.predict(support_set)
    # print('support embedding',support_embeddings)
    prototypes = []

    for label in np.unique(support_labels):
        class_embeddings = support_embeddings[support_labels == label]
        class_prototype = np.mean(class_embeddings, axis=0)
        prototypes.append(class_prototype)
    print('prototype',len(prototypes))
    return np.array(prototypes)


# 定义分类查询样本的辅助函数
def classify_query_samples(query_set, prototypes, model):
    query_embeddings = model.predict(query_set)
    dists = cdist(query_embeddings, prototypes)
    pred_labels = np.argmin(dists, axis=1)
    return pred_labels


# 定义合并新的类原型到旧的类原型中的辅助函数
def update_prototypes(old_prototypes, new_prototypes):
    return np.concatenate((old_prototypes, new_prototypes), axis=0)


# 定义执行Few-shot Learning任务的函数
def perform_few_shot_learning(model, data, labels, n_way, k_shot, n_query, offset):
    support_set, support_labels, query_set, query_labels = generate_support_and_query_sets(data, labels, n_way, k_shot,
                                                                                           n_query)

    prototypes = compute_prototypes(support_set, support_labels, model)
    print('prototypes in perform',len(prototypes))
    pred_labels = classify_query_samples(query_set, prototypes, model)
    pred_labels= pred_labels+offset
    print(pred_labels)
    accuracy = accuracy_score(query_labels, pred_labels)
    print('query_labels:',query_labels)
    print('pred_labels:',pred_labels)
    print(f'Few-shot learning accuracy: {accuracy:.4f}')
    return prototypes, accuracy

from keras.layers import MaxPool1D,Reshape,Flatten
# 定义创建模型的函数
def create_model(input_shape):
    filter = 32
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(filter, 32, activation='relu', input_shape=(input_shape, 1), padding="same"),
        MaxPool1D(pool_size=3, strides=3),
        Conv1D(filter, 32, strides=1, activation='relu', padding='same'),
        MaxPool1D(pool_size=3, strides=3),
        # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),
        # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),
        Conv1D(filter, 32, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(6, activation='softmax')  # 假设我们有10个类别
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_model2(input_shape):
    filter = 128
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(filter, 64, activation='relu', input_shape=(input_shape, 1), padding="same"),
        MaxPool1D(pool_size=3, strides=3),
        Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
        MaxPool1D(pool_size=3, strides=3),
        # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),
        # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),
        Conv1D(filter, 32, strides=1, activation='relu', padding='same'),
        # MaxPool1D(pool_size=3, strides=3),
        GlobalAveragePooling1D(),
        Flatten(),
        # Dense(128, activation='relu'),
        # Dense(128, activation='relu'),
        Dense(6, activation='softmax')  # 假设我们有10个类别
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
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
# def compute_prototypes(support_set, support_labels, model):
#     support_embeddings = model.predict(support_set)
#     prototypes = []
#
#     for label in np.unique(support_labels):
#         class_embeddings = support_embeddings[support_labels == label]
#         class_prototype = np.mean(class_embeddings, axis=0)
#         prototypes.append(class_prototype)
#
#     return np.array(prototypes)


def save_prototypes(prototypes, filename):
    with open(filename, 'wb') as file:
        pickle.dump(prototypes, file)


def load_prototypes(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
from keras import Model
def perform_few_shot_learning2(model, data, labels, n_way, k_shot, n_query):
    support_set, support_labels, query_set, query_labels = generate_support_and_query_sets(data, labels, n_way, k_shot, n_query)
    prototypes = compute_prototypes(support_set, support_labels, model)
    pred_labels = classify_query_samples(query_set, prototypes, model)

    accuracy = accuracy_score(query_labels, pred_labels)
    print(f'Few-shot learning accuracy: {accuracy:.4f}')
    return prototypes, accuracy


def Fourthdataset(x_test5,y_test5,pname5,new_prototypes,cmName):
    # updated_prototypes = update_prototypes(previous_prototypes, new_prototypes)
    #
    # # 保存更新后的类原型
    # save_prototypes(updated_prototypes, protoName)

    # 假设我们有新的查询集数据new_query_set
    # new_query_set = np.random.rand(75, 100, 1)  # 示例查询集数据

    # 对新的查询集进行分类
    pred_labels = classify_query_samples(x_test5, new_prototypes, feature_extractor_model)
    from sklearn.metrics import accuracy_score, confusion_matrix
    print(pred_labels)
    #pred_labels=pred_labels[pred_labels == 3] = 6
    # for i in range(len(pred_labels)):
    #     if pred_labels[i] not in y_test5:
    #         if pred_labels[i] < 4:
    #             pred_labels[i] = np.max(y_test5)
    #         else:
    #             pred_labels[i] = np.min(y_test5)
    print(len(np.unique(pred_labels)))
    print(y_test5)
    print(len(np.unique(y_test5)))
    score = accuracy_score(y_test5, pred_labels)
    cm = confusion_matrix(y_test5, pred_labels)
    print(cm)
    PN2 = []
    for item in pname5:
        if item not in PN2:
            PN2.append(item)
    print(PN2)
    utils.plot_confusion_matrix(cm, PN2, cmName)

    print(score)

def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=stride, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

from  keras.layers import MaxPooling1D,Dropout
def build_resnet(input_shape, num_classes,filters):
    # inputs=Reshape((input_shape, 1), input_shape=(input_shape,)),
    # inputs = Input(shape=(input_shape,))
    # x = Reshape((input_shape, 1))(inputs)
    inputs = Input(shape=(input_shape,))  # 输入形状为 (2000,)
    reshaped = Reshape((input_shape, 1))(inputs)  # 重塑为 (2000, 1)


    x = Conv1D(64, 7, strides=2, padding='same')(reshaped)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 添加残差块
    x = resnet_block(x, 32)
    x = resnet_block(x, 64)
    x = resnet_block(x, 32)


    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x= Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


from FTIR_ReaddataFrom500C4 import readFromPlastics500
if __name__ == '__main__':

    # firstData,secondData,thirdData,pid1,pid2,pid3,pname1,pname2,pname3,wavenumber=get_data()
    #
    # #x_train1,y_train1,x_test1,y_test1=dataAugmenation(firstData,pid1,wavenumber,pname1,1)
    # # fileName='FTIR_PLastics500_c4.csv'
    wavenumber4, forthData, pid4, pname4 = readFromPlastics500('FTIR_PLastics500_c4.csv')
    # wavenumber5, fifthData, pid5, pname5 = readFromPlastics500('FTIR_PLastics500_c8.csv')
    #print(forthData.shape)
    x_train1, y_train1, x_test1, y_test1 = dataAugmenation3(forthData, pid4, wavenumber4, pname4, 1)
    #x_train1, x_test1, y_train1, y_test1= train_test_split(forthData, pid4, test_size=0.7, random_state=1)
    fileName = '4th_Dataset_Resnet'
    print(x_train1.shape)
    # input_shape = (, )
    # input_shape=(len(x_train1[0]),1)
    num_classes = 6
    filters=64
    model = build_resnet(len(x_train1[0]), num_classes,filters)
    # model = create_model(len(x_train1[0]))
    # #
    model.fit(x_train1, y_train1, epochs=100, batch_size=512,validation_split=0.2)
    #
    model.save('Original_modelResnet.h5')

    feature_extractor_model = Model(model.inputs, model.layers[-2].output)
    # #
    # # #
    #
    prototyes=compute_prototypes(x_train1,y_train1,feature_extractor_model)
    extrModelName='model/feature_extractor'+fileName+'_model.h5'
    feature_extractor_model.save(extrModelName)
    extrModelName='model/feature_extractor'+fileName+'_model.h5'
    #extrModelName = 'feature_extractor_model.h5'
    feature_extractor_model = load_model(extrModelName)
    # prototyes = compute_prototypes(x_train1, y_train1, feature_extractor_model)
    # def create_model(input_shape):
    #     filter = 32
    #     model = Sequential([
    #         Reshape((input_shape, 1), input_shape=(input_shape,)),
    #         Conv1D(filter, 32, activation='relu', input_shape=(input_shape, 1), padding="same"),
    #         MaxPool1D(pool_size=3, strides=3),
    #         Conv1D(filter, 32, strides=1, activation='relu', padding='same'),
    #         MaxPool1D(pool_size=3, strides=3),
    #         # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
    #         # MaxPool1D(pool_size=3, strides=3),
    #         # Conv1D(filter, 64, strides=1, activation='relu', padding='same'),
    #         # MaxPool1D(pool_size=3, strides=3),
    #         Conv1D(filter, 32, strides=1, activation='relu', padding='same'),
    #         # MaxPool1D(pool_size=3, strides=3),
    #
    #         Flatten(),
    #         Dense(128, activation='relu'),
    #         Dense(128, activation='relu'),
    #         Dense(6, activation='softmax')  # 假设我们有10个类别
    #     ])
    #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model
    # # 加载预训练模型


    # feature_extractor_model = load_model('feature_extractor_model.h5')
    feature_extractor_model.summary()
    # previous_prototypes = prototyes
    # 加载之前保存的类原型
    # previous_prototypes = load_prototypes(protoName)
    # print('previous_prototypes',len(previous_prototypes))
    # # 假设我们有新的支撑集数据new_support_set和对应的标签new_support_labels
    # new_support_set = np.random.rand(50, 100, 1)  # 示例支撑集数据
    # new_support_labels = np.random.randint(0, 10, 50)  # 示例支撑集标签
   #  pidM2=[]
   # # print(max(pid4))
   #  for i in range(len(pid2)):
   #      pidM2.append(pid2[i]+len(previous_prototypes))
   #
   #  # pidM2=numpy.array(pidM2)
   #
   #  print(pidM2)
   #  pidM4 = []
   #  pid4=np.array(pid4)
   #  # print(max(pid4))
   #  for i in range(len(pid4)):
   #      pidM4.append(pid4[i] + len(previous_prototypes))
   #
   #  # pidM2=numpy.array(pidM2)
   #
   #  print(pidM4)
    x_train2, y_train2, x_test2, y_test2 = dataAugmenation2(secondData, pid2, wavenumber, pname2, 2)

    # 计算新的类原型
    #x_train2, x_test2, y_train2, y_test2 = train_test_split(secondData, pid2, test_size=0.3, random_state=1)
    #x_train2, x_test2, y_train2, y_test2 = train_test_split(secondData, pidM2, test_size=0.7, random_state=1)
    # new_prototypes, _ = perform_few_shot_learning(feature_extractor_model, x_train2, y_train2, n_way=4, k_shot=30,
    #                                               n_query=70,10)
    # new_prototypes,acc= perform_few_shot_learning(feature_extractor_model, x_train2, y_train2,
    #                                               n_way=4, k_shot=5, n_query=5,offset=len(previous_prototypes))

    #x_train4, x_test4, y_train4, y_test4 = train_test_split(forthData, pid4, test_size=0.3, random_state=1)
    x_train5, x_test5, y_train5, y_test5 = train_test_split(fifthData, pid5, test_size=0.3, random_state=1)
    # new_prototypes,acc= perform_few_shot_learning2(feature_extractor_model, x_train5, y_train5,
    #                                               n_way=6, k_shot=10, n_query=10)
    new_prototypes, acc = perform_few_shot_learning2(feature_extractor_model, x_train5, y_train5,
                                                   n_way=6, k_shot=10, n_query=10)


    # new_prototypes, _ = perform_few_shot_learning(feature_extractor_model, x_train2, y_train2, n_way=4, k_shot=30,
    #                                               n_query=70, 10)
    # 更新旧的类原型
    Fourthdataset(x_test5,y_test5,pname5,new_prototypes,'5th dataset')
    new_prototypes2, acc = perform_few_shot_learning2(feature_extractor_model, x_train2, y_train2,
                                                     n_way=4, k_shot=5, n_query=20)

    Fourthdataset(x_test2, y_test2, pname2, new_prototypes2, '2nd dataset')

    # updated_prototypes = update_prototypes(previous_prototypes, new_prototypes)
    #
    # # 保存更新后的类原型
    # save_prototypes(updated_prototypes, protoName)
    #
    # # 假设我们有新的查询集数据new_query_set
    # # new_query_set = np.random.rand(75, 100, 1)  # 示例查询集数据
    #
    # # 对新的查询集进行分类
    # pred_labels = classify_query_samples(x_test2, updated_prototypes, feature_extractor_model)
    # from sklearn.metrics import accuracy_score,confusion_matrix
    # print(pred_labels)
    # #pred_labels=pred_labels[pred_labels == 3] = 6
    # for i in range(len(pred_labels)):
    #     if pred_labels[i] not in y_test2:
    #         if pred_labels[i] < 4:
    #             pred_labels[i] = np.max(y_test2)
    #         else:
    #             pred_labels[i]=np.min(y_test2)
    # print(len(np.unique(pred_labels)))
    # print(y_test2)
    # print(len(np.unique(y_test2)))
    # score=accuracy_score(y_test2,pred_labels)
    # cm= confusion_matrix(y_test2,pred_labels)
    # print(cm)
    # PN2=[]
    # for item in pname2:
    #     if item not in PN2:
    #         PN2.append(item)
    # print(PN2)
    # utils.plot_confusion_matrix(cm,PN2,'Second dataset')
    #
    #
    # print(score)

