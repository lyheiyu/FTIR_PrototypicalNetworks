from utils import utils
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
高斯列主消元算法
'''
# 得到增广矩阵
def get_augmented_matrix(matrix, b):
    row, col = np.shape(matrix)
    matrix = np.insert(matrix, col, values=b, axis=1)
    return matrix
# 取出增广矩阵的系数矩阵（第一列到倒数第二列）
def get_matrix(a_matrix):
    return a_matrix[:, :a_matrix.shape[1] - 1]
# 选列主元，在第k行后的矩阵里，找出最大值和其对应的行号和列号
def get_pos_j_max(matrix, k):
    max_v = np.max(matrix[k:, :])
    pos = np.argwhere(matrix == max_v)
    i, _ = pos[0]
    return i, max_v
# 矩阵的第k行后，行交换
def exchange_row(matrix, r1, r2, k):
    matrix[[r1, r2], k:] = matrix[[r2, r1], k:]
    return matrix
# 消元计算(初等变化)
def elimination(matrix, k):
    row, col = np.shape(matrix)
    for i in range(k + 1, row):
        m_ik = matrix[i][k] / matrix[k][k]
        matrix[i] = -m_ik * matrix[k] + matrix[i]
    return matrix
# 回代求解
def backToSolve(a_matrix):
    matrix = a_matrix[:, :a_matrix.shape[1] - 1]  # 得到系数矩阵
    b_matrix = a_matrix[:, -1]  # 得到值矩阵
    row, col = np.shape(matrix)
    x = [None] * col  # 待求解空间X
    # 先计算上三角矩阵对应的最后一个分量
    x[-1] = b_matrix[col - 1] / matrix[col - 1][col - 1]
    # 从倒数第二行开始回代x分量
    for _ in range(col - 1, 0, -1):
        i = _ - 1
        sij = 0
        xidx = len(x) - 1
        for j in range(col - 1, i, -1):
            sij += matrix[i][j] * x[xidx]
            xidx -= 1
        x[xidx] = (b_matrix[i] - sij) / matrix[i][i]
    return x
# 求解非齐次线性方程组：ax=b
def solve_NLQ(a, b):
    a_matrix = get_augmented_matrix(a, b)
    for k in range(len(a_matrix) - 1):
        # 选列主元
        max_i, max_v = get_pos_j_max(get_matrix(a_matrix), k=k)
        # 如果A[ik][k]=0，则矩阵奇异退出
        if a_matrix[max_i][k] == 0:
            print('矩阵A奇异')
            return None, []
        if max_i != k:
            a_matrix = exchange_row(a_matrix, k, max_i, k=k)
        # 消元计算
        a_matrix = elimination(a_matrix, k=k)
    # 回代求解
    X = backToSolve(a_matrix)
    return a_matrix, X
'''
最小二乘法多项式拟合曲线
'''
# 生成带有噪点的待拟合的数据集合
def last_square_current_loss(xs, ys, A):
    error = 0.0
    for i in range(len(xs)):
        y1 = 0.0
        for k in range(len(A)):
            y1 += A[k] * xs[i] ** k
        error += (ys[i] - y1) ** 2
    return error
def last_square_fit_curve_Gauss(xs, ys, order):
    X, Y = [], []
    # 求解偏导数矩阵里，含有xi的系数矩阵X
    for i in range(0, order + 1):
        X_line = []
        for j in range(0, order + 1):
            sum_xi = 0.0
            for xi in xs:
                sum_xi += xi ** (j + i)
            X_line.append(sum_xi)
        X.append(X_line)
    # 求解偏导数矩阵里，含有yi的系数矩阵Y
    for i in range(0, order + 1):
        Y_line = 0.0
        for j in range(0, order + 1):
            sum_xi_yi = 0.0
            for k in range(len(xs)):
                sum_xi_yi += (xs[k] ** i * ys[k])
            Y_line = sum_xi_yi

        Y.append(Y_line)
    a_matrix, A = solve_NLQ(np.array(X), Y)  # 高斯消元：求解XA=Y的A
    #A = np.linalg.solve(np.array(X), np.array(Y))  # numpy API 求解XA=Y的A
    error = last_square_current_loss(xs=xs, ys=ys, A=A)

    print('最小二乘法+求解线性方程组，误差下降为：{}'.format(error))
    return A
def solve_NLQ(a, b):
    a_matrix = get_augmented_matrix(a, b)
    for k in range(len(a_matrix) - 1):
        # 选列主元
        max_i, max_v = get_pos_j_max(get_matrix(a_matrix), k=k)
        # 如果A[ik][k]=0，则矩阵奇异退出
        if a_matrix[max_i][k] == 0:
            print('矩阵A奇异')
            return None, []
        if max_i != k:
            a_matrix = exchange_row(a_matrix, k, max_i, k=k)
        # 消元计算
        a_matrix = elimination(a_matrix, k=k)
    # 回代求解
    X = backToSolve(a_matrix)
    return a_matrix, X
def draw_fit_curve(xs, A, order,intensity):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #fit_xs, fit_ys = np.arange(min(xs) , max(xs) , 0.01), []

    fit_ys=[]
    fit_xs=xs
    # for i in range(len(A)):
    #     A[i]=random.randint(10,20)*0.7*A[i]
    for i in range(0, len(fit_xs)):
        y = 0.0
        for k in range(0, order + 1):
            y += (A[k] * fit_xs[i] ** k)
        # A[0]=random.randint(0,int(A[0]))

        fit_ys.append(y)
    fit_ys=np.array(fit_ys)
    # resid=ys-fit_ys


    # ax.plot(fit_xs, fit_ys, color='g', linestyle='-', marker='', label='poly-fit')
    # ax.plot(xs, ys, color='m', linestyle='', marker='.', label='true label')
    # #plt.title(s='least square'.format(order))
    # plt.legend()
    # plt.show()
    return fit_xs,fit_ys
def init_fx_data():
    # 待拟合曲线f(x) = sin2x * [(x^2 - 1)^3 + 0.5]
    xs = np.arange(-1, 1, 0.01)  # 200个点
    ys = [((x ** 2 - 1) ** 3 + 0.5) * np.sin(x * 2) for x in xs]
    ys1 = []
    for i in range(len(ys)):
        z = np.random.randint(low=-10, high=10) / 100  # 加入噪点
        ys1.append(ys[i] + z)
    return xs, ys1
polymerName2, waveLength2, intensity2, polymerID2=utils.parseDataForSecondDataset('dataset/new_SecondDataset2.csv')
# polymerName, waveLength, intensity, polymerID = utils.parseData4th('dataset/FourthdatasetFoppl-e.csv')
#polymerName, waveLength, intensity, polymerID = utils.parseData4th('dataset/FourthDataset4.csv')
#polymerName, waveLength, intensity1, polymerID1 = utils.parseData4th('dataset/FourthdatasetFollp-r.csv')
# polymerName, waveLength, intensity, polymerID = utils.parseData4th('dataset/FourthDataset3.csv')
polymerName1, waveLength1, intensity1, polymerID1, x_each, y_each = utils.parseData11('dataset/D4_4_publication11.csv', 2, 1763)
aimNumber=1000

nstep=5
m=int(len(waveLength2)/nstep)
fitnum=int(aimNumber/m)

restnum=aimNumber-fitnum
print(fitnum)
print(fitnum*m)
print(m)

waveLength = np.array(waveLength2)
if waveLength[0] > waveLength[-1]:
    rng = waveLength[0] - waveLength[-1]
else:
    rng = waveLength[-1] - waveLength[0]
half_rng = rng / 2
normalized_wns = (waveLength - np.mean(waveLength)) / half_rng
stepNum=2000
Waveforsteps=np.linspace(max(normalized_wns),min(normalized_wns),stepNum)
order=2
intensitymodify=[]
for item in intensity2[:3]:
    wavetotal = np.zeros(0)
    intensitytotal = np.zeros(0)
    restotal = np.zeros(0)
    for i in range(int(stepNum/nstep)):
        wavestep=np.zeros(nstep)
        intensitystep=np.zeros(nstep)
        # if i>=m-1:
        #     wavestep = normalized_wns[i * nstep:]
        #     intensitystep = item[i * nstep:]
        #
        # else:
        #     wavestep=normalized_wns[i*nstep:(i+1)*nstep]
        #     intensitystep=item[i*nstep:(i+1)*nstep]
        wavestep=Waveforsteps[i*nstep:(i+1)*nstep]
        print(wavestep)
        indexforintensitMax=np.where(normalized_wns>min(wavestep))
        MaxindexList=np.array(indexforintensitMax)
        print(MaxindexList.max())
        indexforintensitmin= np.where(normalized_wns <max(wavestep))
        MinindexList = np.array(indexforintensitmin)
        intensitystep = item[MinindexList.min():MaxindexList.max()]
        print(intensitystep)
        #print(wavestep,intensitystep)
        A = last_square_fit_curve_Gauss(xs=wavestep, ys=intensitystep, order=order)
        wavestep,intensitystep=draw_fit_curve(xs=wavestep, A=A, order=order, intensity=item)
        wavetotal=np.concatenate((wavetotal, wavestep), axis=0)
        print(wavetotal.shape)
        intensitytotal = np.concatenate((intensitytotal, intensitystep), axis=0)
        print(len(intensitytotal))
        # restotal=np.concatenate((restotal,res),axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(wavetotal, intensitytotal, color='g', linestyle='-', marker='', label='poly-fit')
    # ax.plot(wavetotal, restotal, color='b', linestyle='-', marker='', label='poly-fit')
    ax.plot(Waveforsteps, intensitytotal, color='m', linestyle='', marker='.', label='true label')
    ax.plot(normalized_wns, item, color='g', linestyle='-', marker='', label='poly-fit')
    # for ti in range(len(x)):
    #     ax.plot(normalized_wns,x[ti], color='r', linestyle='-', marker='.', label='x label')
    ax.plot(Waveforsteps, intensitytotal, color='r', linestyle='-', marker='.', label='x label')
    plt.show()