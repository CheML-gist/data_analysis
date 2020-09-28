import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#データ読み込み---------------------------------------------------------
sample_data = pd.read_csv('sample.csv', skiprows = 4, header = None)
x = sample_data.iloc[:, 0]
y = sample_data.iloc[:, 1]

#関数の定義------------------------------------------------------------
def gaussian_func(x, mu, sigma, A, s):
    g = np.exp((-(x-mu)**2)/(2*sigma**2))
    g_scaled = A * g + s
    
    return g_scaled

#パラメータ初期値の設定---------------------------------------------------
mu0 = 532.5
sigma0 = 4
A0 = 1400
y0 = 0
initial_param = [mu0, sigma0, A0, y0]

#フィッティング------------------------------------------------------------
p_opt, p_cov = curve_fit(gaussian_func, x, y, p0 = initial_param)

#フィッティイグされた係数とその95%信頼区間の計算---------------------------------
var = np.diag(p_cov)
stdev = np.sqrt(var)
CI_95 = 2 * stdev
FWHM = 2 * np.sqrt(2 * np.log(2)) * p_opt[1]

#フィッテイングデータのプロット-------------------------------------------------
xd = np.linspace(min(x), max(x), 500)
estimated = gaussian_func(xd, p_opt[0], p_opt[1], p_opt[2], p_opt[3])
plt.plot(xd, estimated, label="Estimated curve", color="red")
plt.scatter(x, y, label="raw data", color="blue")
plt.xlabel("Binding energy / eV")
plt.ylabel("Counts")
plt.xlim(525, 540)
plt.legend()
plt.savefig('fit-plot.png', Transparent=True, bbox_inches="tight")
plt.show()

#決定係数の計算--------------------------------------------------------
def estimated_value(x_raw):
    return gaussian_func(x_raw, p_opt[0], p_opt[1], p_opt[2], p_opt[3])

error_sum = 0

for i in range(len(x)):
    error = (y[i] - estimated_value(x[i]))**2
    error_sum = error_sum + error

R2 = 1 - (error_sum / (np.var(y) * (len(x)-1)))    

#各種パラメーターの表示----------------------------------------------------
p_opt = p_opt.astype('float16')
CI_95 = CI_95.astype('float16')

estimated_params = pd.DataFrame(
    [[p_opt[0], p_opt[1], p_opt[2], p_opt[3], FWHM, R2], [CI_95[0], CI_95[1], CI_95[2], CI_95[3]]]
    ).T

estimated_params.index = ['mu', 'sigma', 'A', 'y0', 'FWHM', 'R2']
estimated_params.columns = ['Value', '95% CI']

print(estimated_params)

#解析結果の出力------------------------------------------------------------
output_data = pd.DataFrame([x, y]).append(pd.DataFrame([xd, estimated])).T
output_data.columns = ['raw_x', 'raw_y', 'estimated x', 'estimated y']

output_data.to_csv('estimated_data.csv', index = False)
estimated_params.to_csv('estimated_params.txt', sep = '\t')

