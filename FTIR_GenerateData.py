from utils import utils
import numpy as np
from scipy import interpolate
from sklearn.model_selection import train_test_split
from typing import Union as U, Tuple as T
import matplotlib.pyplot as plt
import pandas as pd
from PLS import airPLS
from sklearn.preprocessing import normalize
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



class GenerateDate:

    def __init__(self):
        pass;

    def readFromPlastics500(self,fileName):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)

        wavenumber = dataset.iloc[1, 3:]
        intensity = dataset.iloc[2:, 3:]
        intensity = np.abs(intensity)
        Pname = dataset.iloc[2:, 1]
        pid = dataset.iloc[2:, 2]
        maxVal = np.max(intensity)

        intensity = maxVal - intensity
        wavenumber = np.array(wavenumber, dtype=np.float)
        intensity = np.array(intensity, dtype=np.float)

        uniqueID = np.unique(pid)
        uniquePN = np.unique(Pname)

        for item in intensity:
            for val in item:
                if val < 0:
                    print(0)
        # intensityBaseline=intensity
        intensityBaseline = []

        for item in intensity:
            item = item - airPLS(item)

            intensityBaseline.append(item)
        intensityForEach = []

        intensityNormalize = []
        # max_val = max(intensityBaseline)
        # def normalize(data, method='max'):
        #     if method == 'max':
        #         max_val = np.max(data)
        #         return data / max_val

        intensityNormalize = normalize(intensityBaseline, 'max')
        intensityBaseline = np.array(intensityBaseline)
        intensityNormalize = np.array(intensityNormalize)

        xnewFinal = np.linspace(max(wavenumber), min(wavenumber), 2000)
        # for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
        for kind in ["cubic"]:  # 插值方式
            # "nearest","zero"为阶梯插值
            # slinear 线性插值
            # "quadratic","cubic" 为2阶、3阶B样条曲线插值
            f = interpolate.interp1d(wavenumber[::-1], intensityNormalize[::-1], kind=kind)
            # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
            # f = interpolate.interp1d(waveLength, intensityDataset1, kind=kind)
            # f4 = interpolate.interp1d(waveLength4, intensityDataset4, kind=kind)

            # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

            ynew = f(xnewFinal)

            return xnewFinal, ynew, pid, Pname

    def find_indices(self,array, max, min):
        indices = [i for i, x in enumerate(array) if x >= min and x <= max]
        return indices

    def findSpectrum(self,spectrum, indexList, num):
        indices = [i for i, x in enumerate(indexList) if x == num]
        FindArray = spectrum[indices]
        return FindArray

    def emsc(self,spectra: np.ndarray, wavenumbers: np.ndarray, order: int = 2,
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

        preprocessed_spectra = (reference + residues / coefs[0]).T

        if return_coefs:
            return preprocessed_spectra, coefs.T

        return preprocessed_spectra
    def dataAugmenation(self,intensity, polymerID, waveLength, pName, randomSeed):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7,
                                                            random_state=randomSeed)

        waveLength = np.array(waveLength, dtype=np.float)
        datas = []
        datas2 = []
        PN = []
        for item in pName:
            if item not in PN:
                PN.append(item)
        polymerMID = []
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
            _, coefs_ = self.emsc(
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
        return x_train, y_train, x_test, y_test

    def dataAugmenation2(self, intensity, polymerID, waveLength, pName, randomSeed):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3,
                                                            random_state=randomSeed)
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
        indicesList = []
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
    def getData(self):

            polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11(
                'dataset/D4_4_publication11.csv',
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
            indices0 = self.find_indices(waveLength, maxwavelength, minwavelenth)
            indices = self.find_indices(waveLength2, maxwavelength, minwavelenth)
            indices4 = self.find_indices(waveLength4, maxwavelength, minwavelenth)
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
            # print('max wavelength', waveLength2[chooseIdex2])
            intensityDataset2 = np.array(intensityDataset2)
            intensityDataset4 = intensityDataset4[::-1]
            for i in range(len(intensityDataset2)):
                intensityDataset2[i] = intensityDataset2[i][::-1]
            # print('intensityDataset2',intensityDataset2.shape)
            waveLength = np.array(waveLength, dtype=np.float)
            # print('wavelength',waveLength)
            waveLength2 = np.array(waveLength2, dtype=np.float)

            waveLength4 = np.array(waveLength4, dtype=np.float)
            # print('wavelength2',waveLength2)
            waveLength = waveLength[chooseIdex01:chooseIdex02]
            # print('wavelength', waveLength)
            waveLength3 = waveLength2[chooseIdex1:chooseIdex2]
            # waveLength3 = waveLength3[::-1]
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
            minwavelenth = max(min(waveLength3), min(waveLength4), min(waveLength))
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
            PPfirstdataset = self.findSpectrum(ynew, polymerID, 3)
            PPseconddataset = self.findSpectrum(ynew2, polymerID2, 3)
            PPthriddataset = self.findSpectrum(ynew3, polymerID4, 9)
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

            return ynew, ynew2, ynew3, polymerID, polymerID2, polymerID4, polymerName, polymerName2, polymerName4, xnewFinal