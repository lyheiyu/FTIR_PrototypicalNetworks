# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
#
# # Simulated FTIR-like data (non-uniform resolution)
#
# from FTIR_ReaddataFrom500C4 import readFromPlastics500
#
# file_path = 'C:/Users/31734/Downloads/FTIR_PrototypicalNetworks-main/FTIR_PrototypicalNetworks-main/FTIR_Dataset/Plastics500C4.csv'
# xnewFinal,ynew,pid,Pname = readFromPlastics500(file_path)
#
# x = np.array([400, 520, 610, 780, 950, 1200, 1500, 1800])
# y = np.array([0.15, 0.32, 0.28, 0.45, 0.40, 0.55, 0.48, 0.60])
#
# # Cubic spline interpolation
# cs = CubicSpline(x, y, bc_type='natural')
#
# # Dense x-grid for smooth curve
# x_dense = np.linspace(x.min(), x.max(), 500)
# y_dense = cs(x_dense)
#
# # Plot
# plt.figure(figsize=(7, 4.5))
# plt.plot(x_dense, y_dense, color='tab:blue', linewidth=2, label='Cubic spline interpolation')
# plt.scatter(x, y, color='black', zorder=3, label='Original data points')
#
# # Optional knot visualization
# for xi in x:
#     plt.axvline(xi, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
#
# plt.xlabel('Wavenumber')
# plt.ylabel('Absorbance')
# plt.legend()
# plt.tight_layout()
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from sympy.printing.pretty.pretty_symbology import line_width

from PLS import airPLS
from sklearn.preprocessing import normalize
def readFromPlastics500_with_plot(fileName, n_points=2000, sample_idx=0, plot=True):


    dataset = pd.read_csv(fileName, header=None, encoding="latin-1", keep_default_na=False)

    # Original grids and data
    wavenumber = dataset.iloc[1, 3:]
    intensity = dataset.iloc[2:, 3:]
    intensity=np.array(intensity,dtype=np.float64)
    intensity = np.abs(intensity)
    Pname = dataset.iloc[2:, 1]
    pid = dataset.iloc[2:, 2]
    maxVal = np.max(intensity)

    intensity = maxVal - intensity
    wavenumber = np.array(wavenumber, dtype=np.float)

    print(wavenumber)
    print(intensity)
    # Baseline correction (airPLS should accept 1D array)
    intensity_baseline = []
    for row in intensity:
        row_bc = row - airPLS(row)
        intensity_baseline.append(row_bc)
    intensity_baseline = np.array(intensity_baseline)             # (N, M)
    intensityNormalize = normalize(intensity_baseline, 'max')
    # Normalization (your normalize should accept 2D)
    # y_raw_all = np.array(normalize(intensity_baseline, "max"))    # (N, M)
    intensityNormalize = np.array(intensityNormalize)
    # Build target grid (FTIR often shown decreasing left->right)
    x_newFinal = np.linspace(wavenumber.max(), wavenumber.min(), n_points)
    print(x_newFinal)
    # x_newFinal = np.linspace(wavenumber.min(), wavenumber.max(), n_points)
    # print(x_newFinal)
    # for i in range(y_raw_all.shape[0]):
    for kind in ["cubic"]:  # 插值方式
        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        f = interpolate.interp1d(wavenumber, intensityNormalize, kind=kind)
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        # f = interpolate.interp1d(waveLength, intensityDataset1, kind=kind)
        # f4 = interpolate.interp1d(waveLength4, intensityDataset4, kind=kind)

        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

        y_new = f(x_newFinal)

        # return x_newFinal, ynew, pid, Pname
    print(y_new.shape)
    print(x_newFinal)
    if plot:
        if sample_idx < 0 or sample_idx >= intensityNormalize.shape[0]:
            raise ValueError(f"sample_idx out of range: 0..{intensityNormalize.shape[0]-1}")

        plt.figure(figsize=(8, 4.8))
        plt.plot(wavenumber, intensityNormalize[sample_idx], label="Original data)",linewidth=1.2,linestyle='-')
        # plt.plot(x_newFinal, y_new[sample_idx], label=f"Cubic spline interpolated ({n_points} points)",linewidth=2.4,linestyle='--')
        plt.plot(x_newFinal, y_new[sample_idx], label=f"Cubic spline interpolated ", linewidth=2.4,
                 linestyle='--')
        plt.xlabel("Wavenumber")
        plt.ylabel("Intensity (normalized)")
        # plt.title(f"PID: {pid[sample_idx]} | Polymer: {Pname[sample_idx]}")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.show()

    return wavenumber, intensityNormalize, x_newFinal, y_new, pid, Pname
if __name__ == '__main__':
    filename='dataset/FTIR_PLastics500_c4.csv'  # Replace with your actual file path
    w_raw, y_raw, x_new, y_new, pid, pname = readFromPlastics500_with_plot(
        filename,
        n_points=2000,
        sample_idx=2,
        plot=True
    )
