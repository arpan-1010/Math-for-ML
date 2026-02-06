import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats

#----------------------------------------------binomial distribution---------------------------------------------#
print(stats.binom.pmf(5, 20, 0.1)) #prob(exactly 5 customers will return the items)

pmf_df = pd.DataFrame({'success' : range(0, 21), 'pmf' : list(stats.binom.pmf(range(0, 21), 20, 0.1))})
sn.barplot(x = pmf_df.success, y = pmf_df.pmf)
plt.ylabel('pmf')
plt.xlabel('Number of items returned')
plt.show()

print(stats.binom.cdf(5, 20, 0.1)) #prob(max of 5 customers will return the item)

print(1 - stats.binom.cdf(5, 20, 0.1)) #prob(more than 5 customers will return the items)

mean, var = stats.binom.stats(20, 0.1)
print('Average : ', mean, 'Variance : ', var)

#----------------------------------------------poisson distribution---------------------------------------------#
print(stats.poisson.cdf(5, 10)) #prob(max 5 calls will arrive at the call center)

print(1 - stats.poisson.cdf(30, 30)) #prob(no of calls over 3hr period will exceed 30)

pmf_df = pd.DataFrame({'success' : range(0, 30), 'pmf' : list(stats.poisson.pmf(range(0, 30), 10))})
sn.barplot(x = pmf_df.success, y = pmf_df.pmf)
plt.ylabel('pmf')
plt.xlabel('Number of items returned')
plt.show()

#----------------------------------------------exponential distribution---------------------------------------------#
print(stats.expon.cdf(1000, loc = 1 / 1000, scale = 1000)) #prob(system will fail before 1000 hrs)

print(1 - stats.expon.cdf(2000, loc = 1 / 1000, scale = 1000)) #prob(system will not fail before 2000 hrs)

print(stats.expon.ppf(0.1, loc = 1 / 1000, scale = 1000)) #time by which 10% of the system will fail

pdf_df = pd.DataFrame({'success' : range(0, 5000, 100), 'pdf' : list(stats.expon.pdf(range(0, 5000, 100), loc = 1 / 1000, scale = 1000))})
plt.figure(figsize = (10, 4))
sn.barplot(x = pdf_df.success, y = pdf_df.pdf)
plt.xticks(rotation = 90)
plt.xlabel('Time to failure')
plt.show()

#----------------------------------------------normal distribution---------------------------------------------#
beml_df = pd.read_csv('BEML.csv')

plt.plot(beml_df.Close)
plt.xlabel('Time')
plt.ylabel('Close')
plt.show()

beml_df['gain'] = beml_df.Close.pct_change(periods = 1)
print(beml_df.head(5))
beml_df = beml_df.dropna()

plt.figure(figsize = (8, 6))
plt.plot(beml_df.index, beml_df.gain)
plt.xlabel('Time')
plt.ylabel('Gain')
plt.show()

#----------------------------------------------mean, variance, std---------------------------------------------#
print("Daily gain of BEML : ")
print("Mean : ", round(beml_df.gain.mean(), 4))
print("Standard Deviation : ", round(beml_df.gain.std(), 4))
print(beml_df.gain.describe())

#-----------------------------------------------confidence interval--------------------------------------------#
import numpy as np

beml_df_ci = stats.norm.interval(0.95, loc = beml_df.gain.mean(), scale = beml_df.gain.std())
print("Gain at 95% confidence interval is : ", np.round(beml_df_ci, 4))

#-----------------------------------------------cumulative prob dist--------------------------------------------#
beml_df_cpd = stats.norm.cdf(-0.02, loc = beml_df.gain.mean(), scale = beml_df.gain.std())
print("Probability of making 2% loss or higher in BEML : ", beml_df_cpd)

beml_df_cpd = 1 - stats.norm.cdf(0.02, loc = beml_df.gain.mean(), scale = beml_df.gain.std())
print("Probability of making 2% gain or higher in BEML : ", beml_df_cpd)






















































