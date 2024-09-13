import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fileName='dataset/FTIR_PLASTIC_c8.csv'
dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        # dataset = dataset.replace({'NaN': pd.np.nan, 'nan': pd.np.nan})
        # dataset= dataset.fillna(0)
polymerName = dataset.iloc[1:, 1]
polymerName=np.array(polymerName)
PN=[]
for item in polymerName:
    if item not in PN:
        PN.append(item)
pid=[]
for i in range(len(polymerName)):
    for j in range(len(PN)):
        if polymerName[i]==PN[j]:

            pid.append(j)
pid=np.array(pid)
firstRaw= dataset.iloc[0, 6:]
firstRaw= np.array(firstRaw)


indicesWave= [l for l, id in enumerate(firstRaw) if id == 'Data(x)']
# indicesIntesity= [l for l, id in enumerate(firstRaw) if id == 'Data(y)']
indicesIntesity= [l for l, id in enumerate(firstRaw) if id == 'Data(Y)']

wavenumber=firstRaw[indicesWave]
dataColum=dataset.iloc[1:,6:]
dataColum=np.array(dataColum)


# print(wavenumber)
wavenumber=dataColum[0][indicesWave]
wavenumber=np.array(wavenumber)
intensity=[]
for i in range(len(dataColum)):
    intensity.append(dataColum[i][indicesIntesity])
    # print(dataColum[i][indicesIntesity])

intensity=np.array(intensity)
print(wavenumber)
print(intensity[0])
# plt.plot(wavenumber,intensity[0])
#
# plt.show()
print(intensity.shape)
first=np.concatenate((0,0),axis=None)
eachRow=[]
eachRow.append(np.concatenate((first,wavenumber),axis=None))
for i in range(len(polymerName)):
    print(polymerName[i],pid[i])
    firstCON=np.concatenate((polymerName[i],pid[i]),axis=None)
    total=np.concatenate((firstCON,intensity[i]),axis=None)
    eachRow.append(total)



eachRow=pd.DataFrame(eachRow)
eachRow.to_csv('FTIR_PLastics500_c8.csv')

# print(eachRow)