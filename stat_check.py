import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# data1 = pd.read_csv('data1.csv')
# data2 = pd.read_csv('data2.csv')
# X  = data1.iloc[:, 0]
# Y1 = data1.iloc[:, 1]
# Y2 = data2.iloc[:, 1]

# data1 = pd.read_csv("ACH_LA_County_income_data.csv")
data1 = pd.read_csv("ACH_LA_County_housing_data.csv")
data2 = pd.read_csv("ACH_LA_County_income_data.csv")
# cols = (0, 11, 25) # year, unemployment pct, income per capita
# X  = data1.iloc[:, cols[0]]
# Y1 = data1.iloc[:, cols[1]] 
# Y1_err = data1.iloc[:, cols[1] + 1] # margin of error
# Y2 = data1.iloc[:, cols[2]] 
# Y2_err = data1.iloc[:, cols[2] + 1] # margin of error

cols = ('year', 
        'TotalHousing_Estimate_Units', 
        'GrossRent_Estimate_MedianDollars',
        'GRAPI_PCT_Over35PCT')
cols2 = ('year', 
         'Population_Estimate_Over16', 
         'Unemployed_PCT_Civilian', 
         'HouseholdIncome_Estimate_Median',
         'Income_Estimate_PerCapita')
        # 'GRAPI_Estimate_30to34p9PCT')
        # 'Income_Estimate_PerCapita') # year, unemployment pct, income per capita

X      = data1.loc[:, cols[0]]
Y1     = data2.loc[:, cols2[3]] 
Y1_err = data2.loc[:, cols2[3] + "_MOE"] # margin of error
Y2     = data1.loc[:, cols[3]] 
Y2_err = data1.loc[:, cols[3] + "_MOE"] # margin of error
# Y3     = data1.loc[:, [cols[2], cols[3]]]
# Y3     = data1.loc[:, cols[3]] 
# Y3_err = data1.loc[:, cols[3] + "_MOE"] # margin of error

fig = plt.figure()

fig1 = fig.add_subplot(311)
# fig1.set_xlabel(data1.columns[cols[0]])
# fig1.set_ylabel(data1.columns[cols[1]])
fig1.set_xlabel(X.name)
fig1.set_ylabel(Y1.name)
model1 = LinearRegression()
model1.fit(X.values.reshape(-1, 1), Y1.values.reshape(-1, 1))
y1_fit = model1.predict(X.values.reshape(-1, 1))
# plt.scatter(X, Y1)
plt.errorbar(X, Y1, yerr=Y1_err, fmt='.')
r2_error1 = r2_score(Y1, y1_fit)
line1 = plt.plot(X, y1_fit, color='red', label="%.4f"%r2_error1)
plt.legend(handles=line1)

fig2 = fig.add_subplot(312)
# fig2.set_xlabel(data1.columns[cols[0]])
# fig2.set_ylabel(data1.columns[cols[2]])
fig2.set_xlabel(X.name)
fig2.set_ylabel(Y2.name)
model2 = LinearRegression()
model2.fit(X.values.reshape(-1, 1), Y2.values.reshape(-1, 1))
y2_fit = model2.predict(X.values.reshape(-1, 1))
# plt.scatter(X, Y2)
plt.errorbar(X, Y2, yerr=Y2_err, fmt='.')
r2_error2 = r2_score(Y2, y2_fit)
line2 = plt.plot(X, y2_fit, color='red', label="%.4f"%r2_error2)
plt.legend(handles=line2)

fig3 = fig.add_subplot(313)
# fig3.set_xlabel(data1.columns[cols[1]])
# fig3.set_ylabel(data1.columns[cols[2]])
fig3.set_xlabel(Y1.name)
fig3.set_ylabel(Y2.name)
model3 = LinearRegression()
model3.fit(Y1.values.reshape(-1, 1), Y2.values.reshape(-1, 1))
y3_fit = model3.predict(Y1.values.reshape(-1, 1))
# plt.scatter(Y1, Y2)
plt.errorbar(Y1, Y2, xerr=Y1_err, yerr=Y2_err, fmt='.')
r2_error3 = r2_score(Y2, y3_fit)
line3 = plt.plot(Y1, y3_fit, color='red', label="%.4f"%r2_error3)
plt.legend(handles=line3)

print('r squared error 1: %.2f' % r2_score(Y1, y1_fit))
print('r squared error 2: %.2f' % r2_score(Y2, y2_fit))
print('r squared error 3: %.2f' % r2_score(Y2, y3_fit))

results = sm.OLS(Y2, y3_fit).fit()
# print(Y3)
# results = sm.OLS(Y1, Y3).fit()
print(results.summary())

# fig = plt.scatter(Y1, Y2)
# fig.set_xlabel(data1.columns[1])
# fig.set_ylabel(data2.columns[1])

plt.show()
