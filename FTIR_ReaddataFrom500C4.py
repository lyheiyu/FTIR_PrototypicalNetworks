import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PLS import airPLS
from sklearn.preprocessing import normalize,MinMaxScaler
from scipy import interpolate


def readFromPlastics500 (fileName):

    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)

    wavenumber=dataset.iloc[1,3:]
    intensity =dataset.iloc[2:,3:]
    intensity= np.abs(intensity)
    Pname= dataset.iloc[2:,1]
    pid= dataset.iloc[2:,2]
    maxVal= np.max(intensity)

    intensity = maxVal - intensity
    wavenumber=np.array(wavenumber,dtype=np.float)
    intensity=np.array(intensity,dtype=np.float)

    uniqueID= np.unique(pid)
    uniquePN= np.unique(Pname)


    for item in intensity:
        for val in item:
            if val<0:
                print(0)
    # intensityBaseline=intensity
    intensityBaseline=[]

    for item in intensity:
        item = item - airPLS(item)

        intensityBaseline.append(item)
    intensityForEach=[]

    intensityNormalize=[]
    # max_val = max(intensityBaseline)
    # def normalize(data, method='max'):
    #     if method == 'max':
    #         max_val = np.max(data)
    #         return data / max_val

    intensityNormalize = normalize(intensityBaseline, 'max')
    intensityBaseline=np.array(intensityBaseline)
    intensityNormalize=np.array(intensityNormalize)

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

        return xnewFinal,ynew,pid,Pname
        # for item in ynew:
        #     pl.plot(xnew2, item)
        # ynew2 = f2(xnew3)
        # ynew3 = f4(xnew4)

    # print(intensityBaseline)
    # print(intensityBaseline[1,2,3,4,5])
    # def normalize(data, method='max'):
    #     if method == 'max':
    #         max_val = max(abs(data.min()), abs(data.max()))
    #         return data / max_val
    # for item in uniqueID:
    #     EachIntensity = [l for l, id in enumerate(pid) if id == item]
    #     print(EachIntensity)
    #     forNormolize=intensityBaseline[EachIntensity]
    #     forNormolize=normalize(forNormolize, 'max')
    #     intensityForEach.append(forNormolize)
# for i in range(len(intensityNormalize)):
#
#         plt.plot(xnewFinal, ynew[i])
#
# plt.show()

# intensity=intensity-airPLS(intensity)

# scaler= StandardScaler().fit(intensityBaseline)
# intensity=scaler.transform(intensityBaseline)
# minScaler = MinMaxScaler().fit(intensityBaseline)
# intensity= minScaler.transform(intensityBaseline)
#
# MMScaler=MinMaxScaler().fit(intensityBaseline)
# intensity=MMScaler.transform(intensityBaseline)
# intensity = normalize(intensityBaseline, 'max')
# def custom_normalize(data):
#     max_val = np.max(data)
#     return data / max_val
#
# intensity = custom_normalize(intensityBaseline)
# min_val = np.min(intensity)
# max_val = np.max(intensity)
# normalized_intensity = (intensity - min_val) / (max_val - min_val)
#
