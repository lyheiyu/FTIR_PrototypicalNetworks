# -*-coding:utf-8 -*-
import numpy
import numpy as np
from scipy import interpolate
import pylab as pl
from  utils import utils
import pandas as pd
def find_indices(array, max,min):
    indices = [i for i, x in enumerate(array) if x >=min and x<=max]
    return indices
def findSpectrum(spectrum,indexList,num):
    indices = [i for i, x in enumerate(indexList) if x==num]
    FindArray=spectrum[indices]
    return FindArray
polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('dataset/D4_4_publication11.csv', 2, 1763)
polymerName2, waveLength2, intensity2, polymerID2 = utils.parseDataForSecondDataset2('dataset/new_SecondDataset2.csv')

for i in range(len(intensity2)):
    intensity2[i]=intensity2[i][::-1]
# waveLength2=waveLength2[::-1]
print(waveLength2)
polymerName4, waveLength4, intensity4, polymerID4 = utils.parseData4th('dataset/FourthdatasetFollp-r3.csv')
for i in range(len(intensity4)):
    intensity4[i]=intensity4[i][::-1]
# for i in range(len(intensity2)):
#     intensity2[i]=intensity2[i][::-1]
# waveLength4=waveLength4[::-1]
intensity3=intensity2[0]
max1=max(waveLength)
max2=max(waveLength2)
max3=max(waveLength4)
min1=min(waveLength)
min2=min(waveLength2)
min3=min(waveLength4)
maxwavelength=min(max1,max2,max3)
minwavelenth=max(min1,min2,min3)
print(maxwavelength,minwavelenth)
indices0=find_indices(waveLength,maxwavelength,minwavelenth)
indices=find_indices(waveLength2,maxwavelength,minwavelenth)
indices4=find_indices(waveLength4,maxwavelength,minwavelenth)
print('4',indices4)
print('0',indices0)
print('1.',indices)
print(len(waveLength))
chooseIdex01=indices0[0]
chooseIdex02=indices0[-1]
chooseIdex1=indices[0]
chooseIdex2=indices[-1]
chooseIdex41=indices4[0]
chooseIdex42=indices4[-1]
print('choose0',chooseIdex01,chooseIdex02)
print('choose4',chooseIdex41,chooseIdex42)
# chooseIdex1=len(waveLength2)-chooseIdex2
# chooseIdex2=len(waveLength2)-chooseIdex1
i=0
ppid=[i for i ,item in enumerate (polymerID2) if item==3 ]
ppid1=[i for i ,item in enumerate (polymerName) if item=='Poly(propylene)' ]
ppid2=[i for i ,item in enumerate (polymerName4) if item=='PP' ]
print('ppid2',ppid2)
# ppid1=[i for i ,item in enumerate (polymerName) if item=='Poly(styrene)' ]

ppidforadd=[]
for item in range(len(ppid)):
    ppidforadd.append(polymerID[ppid1][0])
ppidforadd3=[]
for item in range(len(ppid2)):
    ppidforadd3.append(polymerID[ppid1][0])
print('ppid3',ppidforadd3)
#
# for i in range(len(ppid)):
#     print(ppid[i])
intensityDataset1=[]
for item in intensity:
    intensityDataset1.append(item[chooseIdex01:chooseIdex02])
intensityDataset2=[]
for item in intensity2:
    intensityDataset2.append(item[chooseIdex1:chooseIdex2])
intensityDataset4=[]
for item in intensity4:
    intensityDataset4.append(item[chooseIdex41:chooseIdex42])
print('max wavelength',waveLength2[chooseIdex2])
intensityDataset2=numpy.array(intensityDataset2)

# print('intensityDataset2',intensityDataset2.shape)
waveLength=np.array(waveLength,dtype=np.float)
# print('wavelength',waveLength)
waveLength2=np.array(waveLength2,dtype=np.float)

waveLength4=np.array(waveLength4,dtype=np.float)
# print('wavelength2',waveLength2)
waveLength=waveLength[chooseIdex01:chooseIdex02]
print('wavelength',waveLength)
# waveLength3=waveLength2[::-1][chooseIdex1:chooseIdex2]
# waveLength3=waveLength3[::-1]
waveLength3=waveLength2[chooseIdex1:chooseIdex2]
print('wavelength3',waveLength3)
# waveLength4=waveLength4[::-1][chooseIdex41:chooseIdex42]
# waveLength4=waveLength4[::-1]
waveLength4=waveLength4[chooseIdex41:chooseIdex42]
waveLength42=waveLength4[chooseIdex41:chooseIdex42]

print('wavelength4',waveLength4)
print('wavelength42',waveLength42)
# print(waveLength.shape)
x=np.linspace(0,10,11)
#x=[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
y=np.sin(x)
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
xnew3=np.linspace(max(waveLength3),min(waveLength3),2000)
xnew4=np.linspace(max(waveLength4),min(waveLength4),2000)
xnew2=np.linspace(max(waveLength),min(waveLength),2000)
xnewFinal = np.linspace(maxwavelength, minwavelenth, 2000)
#for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
for kind in [ "cubic"]:  # 插值方式
    #"nearest","zero"为阶梯插值
    #slinear 线性插值
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    f2=interpolate.interp1d(waveLength3,intensityDataset2,kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    f = interpolate.interp1d(waveLength, intensityDataset1, kind=kind)
    f4 = interpolate.interp1d(waveLength4, intensityDataset4, kind=kind)

    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

    ynew = f(xnew2)
    # for item in ynew:
    #     pl.plot(xnew2, item)
    ynew2=f2(xnew3)
    ynew3= f4(xnew4)
    print('ynew2shape',ynew2.shape)
    print('ynew3shape', ynew3.shape)
#     for item in ynew2:
#
#         pl.plot(xnew3,item)
#
# pl.legend(loc="lower right")
# pl.show()
PPfirstdataset=findSpectrum(ynew,polymerID,3)
PPseconddataset=findSpectrum(ynew2,polymerID2,3)
PPthriddataset=findSpectrum(ynew3,polymerID4,9)
PPfirstdataset=np.array(PPfirstdataset)
PPseconddataset=np.array(PPseconddataset)
PPthriddataset=np.array(PPthriddataset)
for i in range(len(PPseconddataset)):
    PPseconddataset[i]=np.flip(PPseconddataset[i])
print('PPfirst',PPfirstdataset.shape)
print('PPsecond',PPseconddataset.shape)
for i in range(len(PPfirstdataset)):
    pl.plot(xnew2, PPfirstdataset[i],'r')
for i in range(len(PPseconddataset)):
    pl.plot(xnew3,PPseconddataset[i],'y')
for i in range(len(PPthriddataset)):
    pl.plot(xnew4,PPthriddataset[i],'b')

pl.show()
ynewforAdd=ynew2[ppid]
ynewforAdd3=ynew3[ppid2]
print('ppidforadd',ppidforadd)
points=1
#second dataset
# Xdataset= np.concatenate((ynew,PPseconddataset[0:points]),axis=0)
# Ydataset= np.concatenate((polymerID,ppidforadd[0:points]),axis=0)
#third dataset
Xdataset= np.concatenate((ynew,PPthriddataset[0:points]),axis=0)
Ydataset= np.concatenate((polymerID,ppidforadd3[0:points]),axis=0)
# Xdataset= ynew
# Ydataset= polymerID
print('Xdataset',Xdataset.shape)
print('Ydataset',Ydataset.shape)
PN=[]
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
for item in polymerName:
    if item not in PN:
        PN.append(item)

cmTotal=np.zeros((len(PN),len(PN)))
m = 0
t_report = []
scoreTotal = np.zeros(5)
for seedNum in range(20):
    # x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=seedNum)
    # x = np.arange(0, 1000, 1)
    #
    # waveLength = np.array(waveLength, dtype=np.float)

    # PN = utils.getPN('data/D4_4_publication11.csv')
    # y_add = []
    # data_plot = []
    # datas=x_train
    # y_adds=y_train

    x_train, x_test, y_train, y_test = train_test_split(Xdataset, Ydataset, test_size=0.3,
                                                        random_state=seedNum)
    waveLength = np.array(waveLength, dtype=np.float)
    pID = []
    for item in y_train:
        if item not in pID:
            pID.append(item)
    if len(pID) < len(PN):
        continue
    Pidtest = []
    for item in y_test:
        if item not in Pidtest:
            Pidtest.append(item)
    if len(Pidtest) < len(PN):
        continue


    # model = KNeighborsClassifier(n_neighbors=2)
    # model=KnnClf.fit(x_train,y_train)
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=2)
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64,64,64), random_state=1)
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
    # model.fit(x_train, y_train)
    # model = KNeighborsClassifier(n_neighbors=2)
    model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo', random_state=0)
    print(x_train.shape)
    print(y_train.shape)

    model = model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    #
    # utils.printScore(y_test, y_pre)
    # #t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # # SVM_report = pd.DataFrame(t)
    # # SVM_report.to_csv('SVM_report5.csv')
    # cm = confusion_matrix(y_test, y_pre)
    #
    # scores = utils.printScore(y_test, y_pre)
    # second dataset using:
    # y_pre = model.predict(PPseconddataset[points:])
    # utils.printScore(ppidforadd[points:], y_pre)
    y_pre = model.predict(PPthriddataset[points:])

    utils.printScore(ppidforadd3[points:], y_pre)
    # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # SVM_report = pd.DataFrame(t)
    # SVM_report.to_csv('SVM_report5.csv')
   # cm = confusion_matrix(y_test, y_pre)
    print(y_pre)
    scores = utils.printScore(ppidforadd3[points:], y_pre)
    # cmTotal = cmTotal + cm
    scoreTotal += scores
    m += 1
    # utils.plot_confusion_matrix(cm, PN, "Spectral network data_SVM")
    # SVM_Confusion_matrix = pd.DataFrame(cm)
    # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
    # modelMLP.fit(x_train, y_train)
    # y_pre = modelMLP.predict(x_test)
    # utils.printScore(y_test, y_pre)
    # PN = utils.getPN('D4_4_publication11.csv')
    # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # cm = confusion_matrix(y_test, y_pre)
    # #utils.plot_confusion_matrix(cm, PN, 'Spectral network data_MLP')
    #
    # modelKNN = KNeighborsClassifier(n_neighbors=3)
    # modelKNN.fit(x_train, y_train)
    # y_pre = modelKNN.predict(x_test)
    # utils.printScore(y_test, y_pre)
    # PN = utils.getPN('D4_4_publication11.csv')
    # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # cm = confusion_matrix(y_test, y_pre)
    print(m)
print(scoreTotal / m)
# maxnumber.append(sum(scoreTotal / m) )
#
# cmTotal = cmTotal / m
# print(cmTotal / m)
# pl.clf()
# utils.plot_confusion_matrix(cmTotal, PN, 'SVM')
# model = KNeighborsClassifier(n_neighbors=2)
# model=KnnClf.fit(x_train,y_train)
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=2)
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64,64,64), random_state=1)
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
# model.fit(x_train, y_train)
# model = KNeighborsClassifier(n_neighbors=2)


# for kind in [ "cubic"]:  # 插值方式
#     #"nearest","zero"为阶梯插值
#     #slinear 线性插值
#     #"quadratic","cubic" 为2阶、3阶B样条曲线插值
#     f=interpolate.interp1d(waveLength,intensity,kind=kind)
#     # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
#
#     ynew=f(xnew2)
#     for item in ynew:
#         pl.plot(xnew2, item)
#     print(ynew)
#
# pl.legend(loc="lower right")
# pl.show()